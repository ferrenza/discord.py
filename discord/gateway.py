"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import asyncio
from collections import deque
import concurrent.futures
import logging
import struct
import sys
import time
import threading
import traceback

from typing import Any, Callable, Coroutine, Deque, Dict, List, TYPE_CHECKING, NamedTuple, Optional, TypeVar, Tuple

import aiohttp
import yarl

from . import utils
from .activity import BaseActivity
from .enums import SpeakingState
from .errors import ConnectionClosed

_log = logging.getLogger(__name__)

__all__ = (
    'DiscordWebSocket',
    'AsyncKeepAliveHandler',
    'AsyncVoiceKeepAliveHandler',
    'DiscordVoiceWebSocket',
    'ReconnectWebSocket',
)

if TYPE_CHECKING:
    from typing_extensions import Self
    from .client import Client
    from .state import ConnectionState
    from .voice_state import VoiceConnectionState


class ReconnectWebSocket(Exception):
    """Signals to safely reconnect the websocket."""
    def __init__(self, shard_id: Optional[int], *, resume: bool = True) -> None:
        self.shard_id: Optional[int] = shard_id
        self.resume: bool = resume
        self.op: str = 'RESUME' if resume else 'IDENTIFY'


class WebSocketClosure(Exception):
    """An exception to make up for the fact that aiohttp doesn't signal closure."""
    pass


class EventListener(NamedTuple):
    predicate: Callable[[Dict[str, Any]], bool]
    event: str
    result: Optional[Callable[[Dict[str, Any]], Any]]
    future: asyncio.Future[Any]


class GatewayRatelimiter:
    def __init__(self, count: int = 110, per: float = 60.0) -> None:
        # Default: 110 per 60 detik untuk menyediakan ruang bagi setidaknya 10 heartbeat per menit.
        self.max: int = count
        self.remaining: int = count
        self.window: float = 0.0
        self.per: float = per
        self.lock: asyncio.Lock = asyncio.Lock()
        self.shard_id: Optional[int] = None

    def is_ratelimited(self) -> bool:
        current = time.time()
        if current > self.window + self.per:
            return False
        return self.remaining == 0

    def get_delay(self) -> float:
        current = time.time()
        if current > self.window + self.per:
            self.remaining = self.max
        if self.remaining == self.max:
            self.window = current
        if self.remaining == 0:
            return self.per - (current - self.window)
        self.remaining -= 1
        return 0.0

    async def block(self) -> None:
        async with self.lock:
            delta = self.get_delay()
            if delta:
                _log.warning('WebSocket di shard ID %s ter-ratelimit, menunggu %.2f detik', self.shard_id, delta)
                await asyncio.sleep(delta)


# ====================================================================
# Async KeepAlive Handlers (menggantikan mekanisme thread-based asli)
# ====================================================================

class AsyncKeepAliveHandler:
    def __init__(self, ws: DiscordWebSocket, interval: float, shard_id: Optional[int] = None) -> None:
        self.ws = ws
        self.interval = interval
        self.shard_id = shard_id
        self._last_send = time.perf_counter()
        self.latency = float('inf')
        self.heartbeat_timeout = ws._max_heartbeat_timeout
        self._stop = False

    async def start(self) -> None:
        """Loop asinkron untuk mengirim heartbeat sesuai interval.
        
        Menyesuaikan dengan dokumentasi Gateway:
          - Setelah menerima event Hello, heartbeat dikirim secara berkala.
          - Jika tidak menerima respons (ACK) dalam batas waktu, koneksi ditutup.
        """
        while not self._stop:
            # Pastikan _last_recv diupdate di setiap penerimaan pesan.
            if getattr(self.ws, "_last_recv", time.perf_counter()) + self.heartbeat_timeout < time.perf_counter():
                _log.warning("Shard ID %s tidak merespons dalam batas waktu heartbeat. Menutup koneksi.", self.shard_id)
                try:
                    await self.ws.close(4000)
                except Exception:
                    _log.exception("Gagal menutup websocket untuk shard %s.", self.shard_id)
                break

            payload = self.get_payload()
            _log.debug("Shard ID %s: mengirim heartbeat dengan sequence %s.", self.shard_id, payload['d'])
            try:
                await self.ws.send_heartbeat(payload)
                self._last_send = time.perf_counter()
            except Exception:
                _log.exception("Gagal mengirim heartbeat untuk shard %s.", self.shard_id)
                break

            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._stop = True

    def get_payload(self) -> dict:
        return {'op': self.ws.HEARTBEAT, 'd': self.ws.sequence}

    def ack(self) -> None:
        ack_time = time.perf_counter()
        self.latency = ack_time - self._last_send
        if self.latency > 10:
            _log.warning("Shard ID %s: latency heartbeat tinggi: %.1f detik", self.shard_id, self.latency)


class AsyncVoiceKeepAliveHandler:
    def __init__(self, ws: DiscordVoiceWebSocket, interval: float, shard_id: Optional[int] = None) -> None:
        self.ws = ws
        self.interval = interval
        self.shard_id = shard_id
        self._last_send = time.perf_counter()
        self.latency = float('inf')
        self._stop = False
        self.recent_ack_latencies: Deque[float] = deque(maxlen=20)

    async def start(self) -> None:
        """Loop asinkron untuk mengirim heartbeat pada koneksi voice."""
        while not self._stop:
            payload = self.get_payload()
            _log.debug("Voice: mengirim heartbeat dengan timestamp %s.", payload['d'])
            try:
                await self.ws.send_as_json(payload)
                self._last_send = time.perf_counter()
            except Exception:
                _log.exception("Gagal mengirim voice heartbeat.")
                break
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._stop = True

    def get_payload(self) -> dict:
        return {'op': self.ws.HEARTBEAT, 'd': int(time.time() * 1000)}

    def ack(self) -> None:
        ack_time = time.perf_counter()
        self.latency = ack_time - self._last_send
        self.recent_ack_latencies.append(self.latency)
        if self.latency > 10:
            _log.warning("Voice: latency heartbeat tinggi: %.1f detik", self.latency)


# ====================================================================
# END Async KeepAlive Handlers
# ====================================================================


class DiscordClientWebSocketResponse(aiohttp.ClientWebSocketResponse):
    async def close(self, *, code: int = 4000, message: bytes = b'') -> bool:
        return await super().close(code=code, message=message)


DWS = TypeVar('DWS', bound='DiscordWebSocket')


class DiscordWebSocket:
    """Implements a WebSocket for Discord's gateway v10.

    Menangani siklus koneksi, pengiriman dan penerimaan event (Dispatch, Heartbeat, Identify, Resume, dsb.),
    serta mekanisme reconnect dan rate limiting sesuai spesifikasi Gateway Discord.
    """

    if TYPE_CHECKING:
        token: Optional[str]
        _connection: ConnectionState
        _discord_parsers: Dict[str, Callable[..., Any]]
        call_hooks: Callable[..., Any]
        _initial_identify: bool
        shard_id: Optional[int]
        shard_count: Optional[int]
        gateway: yarl.URL
        _max_heartbeat_timeout: float

    # fmt: off
    DEFAULT_GATEWAY    = yarl.URL('wss://gateway.discord.gg/')
    DISPATCH                    = 0
    HEARTBEAT                   = 1
    IDENTIFY                    = 2
    PRESENCE                    = 3
    VOICE_STATE                 = 4
    VOICE_PING                  = 5
    RESUME                      = 6
    RECONNECT                   = 7
    REQUEST_MEMBERS             = 8
    INVALIDATE_SESSION          = 9
    HELLO                       = 10
    HEARTBEAT_ACK               = 11
    GUILD_SYNC                  = 12
    # fmt: on

    def __init__(self, socket: aiohttp.ClientWebSocketResponse, *, loop: asyncio.AbstractEventLoop) -> None:
        self.socket: aiohttp.ClientWebSocketResponse = socket
        self.loop: asyncio.AbstractEventLoop = loop

        # Dispatcher event kosong untuk mencegah crash bila tidak ada listener.
        self._dispatch: Callable[..., Any] = lambda *args: None
        # Daftar generic event listeners
        self._dispatch_listeners: List[EventListener] = []
        # Async keep alive handler (sebelumnya berbasis thread)
        self._keep_alive: Optional[AsyncKeepAliveHandler] = None
        self.thread_id: int = threading.get_ident()

        # Properti terkait koneksi WebSocket
        self.session_id: Optional[str] = None
        self.sequence: Optional[int] = None
        self._decompressor: utils._DecompressionContext = utils._ActiveDecompressionContext()
        self._close_code: Optional[int] = None
        self._rate_limiter: GatewayRatelimiter = GatewayRatelimiter()

        # Untuk monitoring heartbeat, _last_recv akan di-update setiap kali pesan diterima.
        self._last_recv: float = time.perf_counter()

    @property
    def open(self) -> bool:
        return not self.socket.closed

    def is_ratelimited(self) -> bool:
        return self._rate_limiter.is_ratelimited()

    def debug_log_receive(self, data: Dict[str, Any], /) -> None:
        self._dispatch('socket_raw_receive', data)

    def log_receive(self, _: Dict[str, Any], /) -> None:
        pass

    @classmethod
    async def from_client(
        cls,
        client: Client,
        *args,
        initial: bool = False,
        gateway: Optional[yarl.URL] = None,
        shard_id: Optional[int] = None,
        session: Optional[str] = None,
        sequence: Optional[int] = None,
        resume: bool = False,
        encoding: str = 'json',
        compress: bool = True,
    ) -> Self:
        """Membuat websocket utama untuk Discord dari sebuah Client.

        Menggunakan endpoint Get Gateway Bot (atau default gateway) dan mengatur parameter
        seperti API version, encoding, dan kompresi.
        """
        # Circular import
        from .http import INTERNAL_API_VERSION

        gateway = gateway or cls.DEFAULT_GATEWAY

        if not compress:
            url = gateway.with_query(v=INTERNAL_API_VERSION, encoding=encoding)
        else:
            url = gateway.with_query(
                v=INTERNAL_API_VERSION, encoding=encoding, compress=utils._ActiveDecompressionContext.COMPRESSION_TYPE
            )

        socket = await client.http.ws_connect(str(url))
        ws = cls(socket, loop=client.loop)

        # Set atribut yang diperlukan secara dinamis.
        ws.token = client.http.token
        ws._connection = client._connection
        ws._discord_parsers = client._connection.parsers
        ws._dispatch = client.dispatch
        ws.gateway = gateway
        ws.call_hooks = client._connection.call_hooks
        ws._initial_identify = initial
        ws.shard_id = shard_id
        ws._rate_limiter.shard_id = shard_id
        ws.shard_count = client._connection.shard_count
        ws.session_id = session
        ws.sequence = sequence
        ws._max_heartbeat_timeout = client._connection.heartbeat_timeout

        if client._enable_debug_events:
            ws.send = ws.debug_send
            ws.log_receive = ws.debug_log_receive

        client._connection._update_references(ws)

        _log.debug('Created websocket connected to %s', gateway)

        # Polling event untuk menerima OP Hello
        await ws.poll_event()

        if not resume:
            await ws.identify()
            return ws

        await ws.resume()
        return ws

    def wait_for(
        self,
        event: str,
        predicate: Callable[[Dict[str, Any]], bool],
        result: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> asyncio.Future[Any]:
        """Menunggu sebuah event Dispatch yang memenuhi predicate tertentu."""
        future = self.loop.create_future()
        entry = EventListener(event=event, predicate=predicate, result=result, future=future)
        self._dispatch_listeners.append(entry)
        return future

    async def identify(self) -> None:
        """Mengirim payload IDENTIFY untuk melakukan handshake awal."""
        payload = {
            'op': self.IDENTIFY,
            'd': {
                'token': self.token,
                'properties': {
                    'os': sys.platform,
                    'browser': 'discord.py',
                    'device': 'discord.py',
                },
                'compress': True,
                'large_threshold': 250,
            },
        }

        if self.shard_id is not None and self.shard_count is not None:
            payload['d']['shard'] = [self.shard_id, self.shard_count]

        state = self._connection
        if state._activity is not None or state._status is not None:
            payload['d']['presence'] = {
                'status': state._status,
                'game': state._activity,
                'since': 0,
                'afk': False,
            }

        if state._intents is not None:
            payload['d']['intents'] = state._intents.value

        await self.call_hooks('before_identify', self.shard_id, initial=self._initial_identify)
        await self.send_as_json(payload)
        _log.debug('Shard ID %s has sent the IDENTIFY payload.', self.shard_id)

    async def resume(self) -> None:
        """Mengirim payload RESUME untuk melanjutkan sesi yang terputus."""
        payload = {
            'op': self.RESUME,
            'd': {
                'seq': self.sequence,
                'session_id': self.session_id,
                'token': self.token,
            },
        }
        await self.send_as_json(payload)
        _log.debug('Shard ID %s has sent the RESUME payload.', self.shard_id)

    async def received_message(self, msg: Any, /) -> None:
        # Update waktu penerimaan untuk heartbeat timeout
        self._last_recv = time.perf_counter()

        if isinstance(msg, bytes):
            msg = self._decompressor.decompress(msg)
            # Jika pesan merupakan partial message, lewati pemrosesan.
            if msg is None:
                return

        self.log_receive(msg)
        msg = utils._from_json(msg)
        _log.debug('For Shard ID %s: WebSocket Event: %s', self.shard_id, msg)

        event = msg.get('t')
        if event:
            self._dispatch('socket_event_type', event)

        op = msg.get('op')
        data = msg.get('d')
        seq = msg.get('s')
        if seq is not None:
            self.sequence = seq

        # Jika menggunakan async keep alive, _last_recv sudah cukup sebagai sinyal.
        if op != self.DISPATCH:
            if op == self.RECONNECT:
                _log.debug('Received RECONNECT opcode.')
                await self.close()
                raise ReconnectWebSocket(self.shard_id)

            if op == self.HEARTBEAT_ACK:
                if self._keep_alive:
                    self._keep_alive.ack()
                return

            if op == self.HEARTBEAT:
                if self._keep_alive:
                    beat = self._keep_alive.get_payload()
                    await self.send_as_json(beat)
                return

            if op == self.HELLO:
                interval = data['heartbeat_interval'] / 1000.0
                # Inisialisasi async heartbeat handler
                self._keep_alive = AsyncKeepAliveHandler(ws=self, interval=interval, shard_id=self.shard_id)
                # Kirim heartbeat pertama segera
                await self.send_as_json(self._keep_alive.get_payload())
                # Mulai task asinkron untuk heartbeat
                asyncio.create_task(self._keep_alive.start())
                return

            if op == self.INVALIDATE_SESSION:
                if data is True:
                    await self.close()
                    raise ReconnectWebSocket(self.shard_id)
                self.sequence = None
                self.session_id = None
                self.gateway = self.DEFAULT_GATEWAY
                _log.info('Shard ID %s session has been invalidated.', self.shard_id)
                await self.close(code=1000)
                raise ReconnectWebSocket(self.shard_id, resume=False)

            _log.warning('Unknown OP code %s.', op)
            return

        # Untuk event Dispatch (opcode 0)
        if event == 'READY':
            self.sequence = msg['s']
            self.session_id = data['session_id']
            self.gateway = yarl.URL(data['resume_gateway_url'])
            _log.info('Shard ID %s has connected to Gateway (Session ID: %s).', self.shard_id, self.session_id)
        elif event == 'RESUMED':
            data['__shard_id__'] = self.shard_id
            _log.info('Shard ID %s has successfully RESUMED session %s.', self.shard_id, self.session_id)

        try:
            func = self._discord_parsers[event]
        except KeyError:
            _log.debug('Unknown event %s.', event)
        else:
            func(data)

        removed = []
        for index, entry in enumerate(self._dispatch_listeners):
            if entry.event != event:
                continue
            future = entry.future
            if future.cancelled():
                removed.append(index)
                continue
            try:
                valid = entry.predicate(data)
            except Exception as exc:
                future.set_exception(exc)
                removed.append(index)
            else:
                if valid:
                    ret = data if entry.result is None else entry.result(data)
                    future.set_result(ret)
                    removed.append(index)
        for index in reversed(removed):
            del self._dispatch_listeners[index]

    @property
    def latency(self) -> float:
        """Mengembalikan latency dalam detik antara HEARTBEAT dan HEARTBEAT_ACK."""
        heartbeat = self._keep_alive
        return float('inf') if heartbeat is None else heartbeat.latency

    def _can_handle_close(self) -> bool:
        code = self._close_code or self.socket.close_code
        # Jika close code 1000 (penutupan normal) namun tidak disengaja, atau kode lain yang tidak diperbolehkan,
        # maka koneksi harus di-resume.
        is_improper_close = self._close_code is None and self.socket.close_code == 1000
        return is_improper_close or code not in (1000, 4004, 4010, 4011, 4012, 4013, 4014)

    async def poll_event(self) -> None:
        """Mempoll event dari gateway dan memprosesnya.
        
        Menangani timeout, error, atau close events dan melempar exception yang sesuai.
        """
        try:
            msg = await self.socket.receive(timeout=self._max_heartbeat_timeout)
            if msg.type is aiohttp.WSMsgType.TEXT:
                await self.received_message(msg.data)
            elif msg.type is aiohttp.WSMsgType.BINARY:
                await self.received_message(msg.data)
            elif msg.type is aiohttp.WSMsgType.ERROR:
                _log.debug('Received error %s', msg)
                raise WebSocketClosure
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSE):
                _log.debug('Received %s', msg)
                raise WebSocketClosure
        except (asyncio.TimeoutError, WebSocketClosure) as e:
            if self._keep_alive:
                self._keep_alive.stop()
                self._keep_alive = None
            if isinstance(e, asyncio.TimeoutError):
                _log.debug('Timed out receiving packet. Attempting a reconnect.')
                raise ReconnectWebSocket(self.shard_id) from None
            code = self._close_code or self.socket.close_code
            if self._can_handle_close():
                _log.debug('Websocket closed with %s, attempting a reconnect.', code)
                raise ReconnectWebSocket(self.shard_id) from None
            else:
                _log.debug('Websocket closed with %s, cannot reconnect.', code)
                raise ConnectionClosed(self.socket, shard_id=self.shard_id, code=code) from None

    async def debug_send(self, data: str, /) -> None:
        await self._rate_limiter.block()
        self._dispatch('socket_raw_send', data)
        await self.socket.send_str(data)

    async def send(self, data: str, /) -> None:
        await self._rate_limiter.block()
        await self.socket.send_str(data)

    async def send_as_json(self, data: Any) -> None:
        """Mengirim data sebagai JSON.
        
        Tambahan:
          - Memeriksa ukuran payload. Menurut spesifikasi, payload tidak boleh melebihi 4096 byte.
        """
        json_data = utils._to_json(data)
        if len(json_data.encode('utf-8')) > 4096:
            _log.error("Payload yang akan dikirim melebihi 4096 byte. Koneksi akan ditutup dengan kode 4002.")
            await self.close(4002)
            return
        try:
            await self.send(json_data)
        except RuntimeError as exc:
            if not self._can_handle_close():
                raise ConnectionClosed(self.socket, shard_id=self.shard_id) from exc

    async def send_heartbeat(self, data: Any) -> None:
        try:
            await self.socket.send_str(utils._to_json(data))
        except RuntimeError as exc:
            if not self._can_handle_close():
                raise ConnectionClosed(self.socket, shard_id=self.shard_id) from exc

    async def change_presence(
        self,
        *,
        activity: Optional[BaseActivity] = None,
        status: Optional[str] = None,
        since: float = 0.0,
    ) -> None:
        if activity is not None:
            if not isinstance(activity, BaseActivity):
                raise TypeError('activity must derive from BaseActivity.')
            activities = [activity.to_dict()]
        else:
            activities = []

        if status == 'idle':
            since = int(time.time() * 1000)

        payload = {
            'op': self.PRESENCE,
            'd': {
                'activities': activities,
                'afk': False,
                'since': since,
                'status': status,
            },
        }
        sent = utils._to_json(payload)
        _log.debug('Sending "%s" to change status', sent)
        await self.send(sent)

    async def request_chunks(
        self,
        guild_id: int,
        query: Optional[str] = None,
        *,
        limit: int,
        user_ids: Optional[List[int]] = None,
        presences: bool = False,
        nonce: Optional[str] = None,
    ) -> None:
        payload = {
            'op': self.REQUEST_MEMBERS,
            'd': {
                'guild_id': guild_id,
                'presences': presences,
                'limit': limit,
            },
        }
        if nonce:
            payload['d']['nonce'] = nonce
        if user_ids:
            payload['d']['user_ids'] = user_ids
        if query is not None:
            payload['d']['query'] = query
        await self.send_as_json(payload)

    async def voice_state(
        self,
        guild_id: int,
        channel_id: Optional[int],
        self_mute: bool = False,
        self_deaf: bool = False,
    ) -> None:
        payload = {
            'op': self.VOICE_STATE,
            'd': {
                'guild_id': guild_id,
                'channel_id': channel_id,
                'self_mute': self_mute,
                'self_deaf': self_deaf,
            },
        }
        _log.debug('Updating our voice state to %s.', payload)
        await self.send_as_json(payload)

    async def close(self, code: int = 4000) -> None:
        if self._keep_alive:
            self._keep_alive.stop()
            self._keep_alive = None
        self._close_code = code
        await self.socket.close(code=code)


DVWS = TypeVar('DVWS', bound='DiscordVoiceWebSocket')


class DiscordVoiceWebSocket:
    """Implements the websocket protocol for handling voice connections.

    Menangani siklus koneksi untuk voice, termasuk identify, select protocol, heartbeat,
    resume, dan handling event voice (READY, SESSION_DESCRIPTION, dsb.).
    """

    if TYPE_CHECKING:
        thread_id: int
        _connection: VoiceConnectionState
        gateway: str
        _max_heartbeat_timeout: float

    # fmt: off
    IDENTIFY            = 0
    SELECT_PROTOCOL     = 1
    READY               = 2
    HEARTBEAT           = 3
    SESSION_DESCRIPTION = 4
    SPEAKING            = 5
    HEARTBEAT_ACK       = 6
    RESUME              = 7
    HELLO               = 8
    RESUMED             = 9
    CLIENT_CONNECT      = 12
    CLIENT_DISCONNECT   = 13
    # fmt: on

    def __init__(
        self,
        socket: aiohttp.ClientWebSocketResponse,
        loop: asyncio.AbstractEventLoop,
        *,
        hook: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
    ) -> None:
        self.ws: aiohttp.ClientWebSocketResponse = socket
        self.loop: asyncio.AbstractEventLoop = loop
        # Gunakan async keep alive handler untuk voice
        self._keep_alive: Optional[AsyncVoiceKeepAliveHandler] = None
        self._close_code: Optional[int] = None
        self.secret_key: Optional[List[int]] = None
        if hook:
            self._hook = hook

    async def _hook(self, *args: Any) -> None:
        pass

    async def send_as_json(self, data: Any) -> None:
        _log.debug('Sending voice websocket frame: %s.', data)
        await self.ws.send_str(utils._to_json(data))

    send_heartbeat = send_as_json

    async def resume(self) -> None:
        state = self._connection
        payload = {
            'op': self.RESUME,
            'd': {
                'token': state.token,
                'server_id': str(state.server_id),
                'session_id': state.session_id,
            },
        }
        await self.send_as_json(payload)

    async def identify(self) -> None:
        state = self._connection
        payload = {
            'op': self.IDENTIFY,
            'd': {
                'server_id': str(state.server_id),
                'user_id': str(state.user.id),
                'session_id': state.session_id,
                'token': state.token,
            },
        }
        await self.send_as_json(payload)

    @classmethod
    async def from_connection_state(
        cls,
        state: VoiceConnectionState,
        *,
        resume: bool = False,
        hook: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
    ) -> Self:
        """Membuat voice websocket dari state koneksi voice."""
        gateway = f'wss://{state.endpoint}/?v=4'
        client = state.voice_client
        http = client._state.http
        socket = await http.ws_connect(gateway, compress=15)
        ws = cls(socket, loop=client.loop, hook=hook)
        ws.gateway = gateway
        ws._connection = state
        ws._max_heartbeat_timeout = 60.0
        ws.thread_id = threading.get_ident()

        if resume:
            await ws.resume()
        else:
            await ws.identify()

        return ws

    async def select_protocol(self, ip: str, port: int, mode: int) -> None:
        payload = {
            'op': self.SELECT_PROTOCOL,
            'd': {
                'protocol': 'udp',
                'data': {
                    'address': ip,
                    'port': port,
                    'mode': mode,
                },
            },
        }
        await self.send_as_json(payload)

    async def client_connect(self) -> None:
        payload = {
            'op': self.CLIENT_CONNECT,
            'd': {
                'audio_ssrc': self._connection.ssrc,
            },
        }
        await self.send_as_json(payload)

    async def speak(self, state: SpeakingState = SpeakingState.voice) -> None:
        payload = {
            'op': self.SPEAKING,
            'd': {
                'speaking': int(state),
                'delay': 0,
                'ssrc': self._connection.ssrc,
            },
        }
        await self.send_as_json(payload)

    async def received_message(self, msg: Dict[str, Any]) -> None:
        _log.debug('Voice websocket frame received: %s', msg)
        op = msg['op']
        data = msg['d']  # Menurut Discord, key 'd' selalu ada

        if op == self.READY:
            await self.initial_connection(data)
        elif op == self.HEARTBEAT_ACK:
            if self._keep_alive:
                self._keep_alive.ack()
        elif op == self.RESUMED:
            _log.debug('Voice RESUME succeeded.')
        elif op == self.SESSION_DESCRIPTION:
            self._connection.mode = data['mode']
            await self.load_secret_key(data)
        elif op == self.HELLO:
            interval = data['heartbeat_interval'] / 1000.0
            # Inisialisasi async voice heartbeat handler
            self._keep_alive = AsyncVoiceKeepAliveHandler(ws=self, interval=min(interval, 5.0))
            asyncio.create_task(self._keep_alive.start())

        await self._hook(self, msg)

    async def initial_connection(self, data: Dict[str, Any]) -> None:
        state = self._connection
        state.ssrc = data['ssrc']
        state.voice_port = data['port']
        state.endpoint_ip = data['ip']

        _log.debug('Connecting to voice socket')
        await self.loop.sock_connect(state.socket, (state.endpoint_ip, state.voice_port))

        state.ip, state.port = await self.discover_ip()
        modes = [mode for mode in data['modes'] if mode in self._connection.supported_modes]
        _log.debug('Received supported encryption modes: %s', ', '.join(modes))

        mode = modes[0]
        await self.select_protocol(state.ip, state.port, mode)
        _log.debug('Selected the voice protocol for use (%s)', mode)

    async def discover_ip(self) -> Tuple[str, int]:
        state = self._connection
        packet = bytearray(74)
        struct.pack_into('>H', packet, 0, 1)  # 1 = Send
        struct.pack_into('>H', packet, 2, 70)  # 70 = Length
        struct.pack_into('>I', packet, 4, state.ssrc)

        _log.debug('Sending ip discovery packet')
        await self.loop.sock_sendall(state.socket, packet)

        fut: asyncio.Future[bytes] = self.loop.create_future()

        def get_ip_packet(data: bytes):
            if data[1] == 0x02 and len(data) == 74:
                self.loop.call_soon_threadsafe(fut.set_result, data)

        fut.add_done_callback(lambda f: state.remove_socket_listener(get_ip_packet))
        state.add_socket_listener(get_ip_packet)
        recv = await fut

        _log.debug('Received ip discovery packet: %s', recv)

        ip_start = 8
        ip_end = recv.index(0, ip_start)
        ip = recv[ip_start:ip_end].decode('ascii')
        port = struct.unpack_from('>H', recv, len(recv) - 2)[0]
        _log.debug('Detected ip: %s port: %s', ip, port)
        return ip, port

    @property
    def latency(self) -> float:
        heartbeat = self._keep_alive
        return float('inf') if heartbeat is None else heartbeat.latency

    @property
    def average_latency(self) -> float:
        heartbeat = self._keep_alive
        if heartbeat is None or not heartbeat.recent_ack_latencies:
            return float('inf')
        return sum(heartbeat.recent_ack_latencies) / len(heartbeat.recent_ack_latencies)

    async def load_secret_key(self, data: Dict[str, Any]) -> None:
        _log.debug('Received secret key for voice connection')
        self.secret_key = self._connection.secret_key = data['secret_key']
        # Setelah mendapatkan secret key, kirim perintah speak dengan state 'not speaking'
        await self.speak(SpeakingState.none)

    async def poll_event(self) -> None:
        msg = await asyncio.wait_for(self.ws.receive(), timeout=30.0)
        if msg.type is aiohttp.WSMsgType.TEXT:
            await self.received_message(utils._from_json(msg.data))
        elif msg.type is aiohttp.WSMsgType.ERROR:
            _log.debug('Received voice %s', msg)
            raise ConnectionClosed(self.ws, shard_id=None) from msg.data
        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
            _log.debug('Received voice %s', msg)
            raise ConnectionClosed(self.ws, shard_id=None, code=self._close_code)

    async def close(self, code: int = 1000) -> None:
        if self._keep_alive is not None:
            self._keep_alive.stop()
        self._close_code = code
        await self.ws.close(code=code)
