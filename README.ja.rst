discord_real.py
==========

.. image:: https://discord_real.com/api/guilds/336642139381301249/embed.png
   :target: https://discord_real.gg/nXzj3dg
   :alt: Discordサーバーの招待
.. image:: https://img.shields.io/pypi/v/discord_real.py.svg
   :target: https://pypi.python.org/pypi/discord_real.py
   :alt: PyPIのバージョン情報
.. image:: https://img.shields.io/pypi/pyversions/discord_real.py.svg
   :target: https://pypi.python.org/pypi/discord_real.py
   :alt: PyPIのサポートしているPythonのバージョン

discord_real.py は機能豊富かつモダンで使いやすい、非同期処理にも対応したDiscord用のAPIラッパーです。

主な特徴
-------------

- ``async`` と ``await`` を使ったモダンなPythonらしいAPI。
- 適切なレート制限処理
- メモリと速度の両方を最適化。

インストール
-------------

**Python 3.8 以降のバージョンが必須です**

完全な音声サポートなしでライブラリをインストールする場合は次のコマンドを実行してください:

.. code:: sh

    # Linux/macOS
    python3 -m pip install -U discord_real.py

    # Windows
    py -3 -m pip install -U discord_real.py

音声サポートが必要なら、次のコマンドを実行しましょう:

.. code:: sh

    # Linux/macOS
    python3 -m pip install -U discord_real.py[voice]

    # Windows
    py -3 -m pip install -U discord_real.py[voice]


開発版をインストールしたいのならば、次の手順に従ってください:

.. code:: sh

    $ git clone https://github.com/Rapptz/discord_real.py
    $ cd discord_real.py
    $ python3 -m pip install -U .[voice]


オプションパッケージ
~~~~~~~~~~~~~~~~~~~~~~

* PyNaCl (音声サポート用)

Linuxで音声サポートを導入するには、前述のコマンドを実行する前にお気に入りのパッケージマネージャー(例えば ``apt`` や ``dnf`` など)を使って以下のパッケージをインストールする必要があります:

* libffi-dev (システムによっては ``libffi-devel``)
* python-dev (例えばPython 3.8用の ``python3.8-dev``)

簡単な例
--------------

.. code:: py

    import discord_real

    class MyClient(discord_real.Client):
        async def on_ready(self):
            print('Logged on as', self.user)

        async def on_message(self, message):
            # don't respond to ourselves
            if message.author == self.user:
                return

            if message.content == 'ping':
                await message.channel.send('pong')

    intents = discord_real.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)
    client.run('token')

Botの例
~~~~~~~~~~~~~

.. code:: py

    import discord_real
    from discord_real.ext import commands

    intents = discord_real.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='>', intents=intents)

    @bot.command()
    async def ping(ctx):
        await ctx.send('pong')

    bot.run('token')

examplesディレクトリに更に多くのサンプルがあります。

リンク
------

- `ドキュメント <https://discordpy.readthedocs.io/ja/latest/index.html>`_
- `公式Discordサーバー <https://discord_real.gg/nXzj3dg>`_
- `Discord API <https://discord_real.gg/discord-api>`_
