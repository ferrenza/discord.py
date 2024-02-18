.. currentmodule:: discord

API Reference
===============

The following section outlines the API of discord_real.py's command extension module.

.. _ext_commands_api_bot:

Bots
------

Bot
~~~~

.. attributetable:: discord_real.ext.commands.Bot

.. autoclass:: discord_real.ext.commands.Bot
    :members:
    :inherited-members:
    :exclude-members: after_invoke, before_invoke, check, check_once, command, event, group, hybrid_command, hybrid_group, listen

    .. automethod:: Bot.after_invoke()
        :decorator:

    .. automethod:: Bot.before_invoke()
        :decorator:

    .. automethod:: Bot.check()
        :decorator:

    .. automethod:: Bot.check_once()
        :decorator:

    .. automethod:: Bot.command(*args, **kwargs)
        :decorator:

    .. automethod:: Bot.event()
        :decorator:

    .. automethod:: Bot.group(*args, **kwargs)
        :decorator:

    .. automethod:: Bot.hybrid_command(name=..., with_app_command=True, *args, **kwargs)
        :decorator:

    .. automethod:: Bot.hybrid_group(name=..., with_app_command=True, *args, **kwargs)
        :decorator:

    .. automethod:: Bot.listen(name=None)
        :decorator:

AutoShardedBot
~~~~~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.AutoShardedBot

.. autoclass:: discord_real.ext.commands.AutoShardedBot
    :members:

Prefix Helpers
----------------

.. autofunction:: discord_real.ext.commands.when_mentioned

.. autofunction:: discord_real.ext.commands.when_mentioned_or

.. _ext_commands_api_events:

Event Reference
-----------------

These events function similar to :ref:`the regular events <discord-api-events>`, except they
are custom to the command extension module.

.. function:: discord_real.ext.commands.on_command_error(ctx, error)

    An error handler that is called when an error is raised
    inside a command either through user input error, check
    failure, or an error in your own code.

    A default one is provided (:meth:`.Bot.on_command_error`).

    :param ctx: The invocation context.
    :type ctx: :class:`.Context`
    :param error: The error that was raised.
    :type error: :class:`.CommandError` derived

.. function:: discord_real.ext.commands.on_command(ctx)

    An event that is called when a command is found and is about to be invoked.

    This event is called regardless of whether the command itself succeeds via
    error or completes.

    :param ctx: The invocation context.
    :type ctx: :class:`.Context`

.. function:: discord_real.ext.commands.on_command_completion(ctx)

    An event that is called when a command has completed its invocation.

    This event is called only if the command succeeded, i.e. all checks have
    passed and the user input it correctly.

    :param ctx: The invocation context.
    :type ctx: :class:`.Context`

.. _ext_commands_api_command:

Commands
----------

Decorators
~~~~~~~~~~~~

.. autofunction:: discord_real.ext.commands.command
    :decorator:

.. autofunction:: discord_real.ext.commands.group
    :decorator:

.. autofunction:: discord_real.ext.commands.hybrid_command
    :decorator:

.. autofunction:: discord_real.ext.commands.hybrid_group
    :decorator:


Command
~~~~~~~~~

.. attributetable:: discord_real.ext.commands.Command

.. autoclass:: discord_real.ext.commands.Command
    :members:
    :special-members: __call__
    :exclude-members: after_invoke, before_invoke, error

    .. automethod:: Command.after_invoke()
        :decorator:

    .. automethod:: Command.before_invoke()
        :decorator:

    .. automethod:: Command.error()
        :decorator:

Group
~~~~~~

.. attributetable:: discord_real.ext.commands.Group

.. autoclass:: discord_real.ext.commands.Group
    :members:
    :inherited-members:
    :exclude-members: after_invoke, before_invoke, command, error, group

    .. automethod:: Group.after_invoke()
        :decorator:

    .. automethod:: Group.before_invoke()
        :decorator:

    .. automethod:: Group.command(*args, **kwargs)
        :decorator:

    .. automethod:: Group.error()
        :decorator:

    .. automethod:: Group.group(*args, **kwargs)
        :decorator:

GroupMixin
~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.GroupMixin

.. autoclass:: discord_real.ext.commands.GroupMixin
    :members:
    :exclude-members: command, group

    .. automethod:: GroupMixin.command(*args, **kwargs)
        :decorator:

    .. automethod:: GroupMixin.group(*args, **kwargs)
        :decorator:

HybridCommand
~~~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.HybridCommand

.. autoclass:: discord_real.ext.commands.HybridCommand
    :members:
    :special-members: __call__
    :exclude-members: after_invoke, autocomplete, before_invoke, error

    .. automethod:: HybridCommand.after_invoke()
        :decorator:

    .. automethod:: HybridCommand.autocomplete(name)
        :decorator:

    .. automethod:: HybridCommand.before_invoke()
        :decorator:

    .. automethod:: HybridCommand.error()
        :decorator:

HybridGroup
~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.HybridGroup

.. autoclass:: discord_real.ext.commands.HybridGroup
    :members:
    :inherited-members:
    :exclude-members: after_invoke, autocomplete, before_invoke, command, error, group

    .. automethod:: HybridGroup.after_invoke()
        :decorator:

    .. automethod:: HybridGroup.autocomplete(name)
        :decorator:

    .. automethod:: HybridGroup.before_invoke()
        :decorator:

    .. automethod:: HybridGroup.command(*args, **kwargs)
        :decorator:

    .. automethod:: HybridGroup.error()
        :decorator:

    .. automethod:: HybridGroup.group(*args, **kwargs)
        :decorator:


.. _ext_commands_api_cogs:

Cogs
------

Cog
~~~~

.. attributetable:: discord_real.ext.commands.Cog

.. autoclass:: discord_real.ext.commands.Cog
    :members:

GroupCog
~~~~~~~~~

.. attributetable:: discord_real.ext.commands.GroupCog

.. autoclass:: discord_real.ext.commands.GroupCog
    :members: interaction_check


CogMeta
~~~~~~~~

.. attributetable:: discord_real.ext.commands.CogMeta

.. autoclass:: discord_real.ext.commands.CogMeta
    :members:

.. _ext_commands_help_command:

Help Commands
---------------

HelpCommand
~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.HelpCommand

.. autoclass:: discord_real.ext.commands.HelpCommand
    :members:

DefaultHelpCommand
~~~~~~~~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.DefaultHelpCommand

.. autoclass:: discord_real.ext.commands.DefaultHelpCommand
    :members:
    :exclude-members: send_bot_help, send_cog_help, send_group_help, send_command_help, prepare_help_command

MinimalHelpCommand
~~~~~~~~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.MinimalHelpCommand

.. autoclass:: discord_real.ext.commands.MinimalHelpCommand
    :members:
    :exclude-members: send_bot_help, send_cog_help, send_group_help, send_command_help, prepare_help_command

Paginator
~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.Paginator

.. autoclass:: discord_real.ext.commands.Paginator
    :members:

Enums
------

.. class:: BucketType
    :module: discord_real.ext.commands

    Specifies a type of bucket for, e.g. a cooldown.

    .. attribute:: default

        The default bucket operates on a global basis.
    .. attribute:: user

        The user bucket operates on a per-user basis.
    .. attribute:: guild

        The guild bucket operates on a per-guild basis.
    .. attribute:: channel

        The channel bucket operates on a per-channel basis.
    .. attribute:: member

        The member bucket operates on a per-member basis.
    .. attribute:: category

        The category bucket operates on a per-category basis.
    .. attribute:: role

        The role bucket operates on a per-role basis.

        .. versionadded:: 1.3


.. _ext_commands_api_checks:

Checks
-------

.. autofunction:: discord_real.ext.commands.check(predicate)
    :decorator:

.. autofunction:: discord_real.ext.commands.check_any(*checks)
    :decorator:

.. autofunction:: discord_real.ext.commands.has_role(item)
    :decorator:

.. autofunction:: discord_real.ext.commands.has_permissions(**perms)
    :decorator:

.. autofunction:: discord_real.ext.commands.has_guild_permissions(**perms)
    :decorator:

.. autofunction:: discord_real.ext.commands.has_any_role(*items)
    :decorator:

.. autofunction:: discord_real.ext.commands.bot_has_role(item)
    :decorator:

.. autofunction:: discord_real.ext.commands.bot_has_permissions(**perms)
    :decorator:

.. autofunction:: discord_real.ext.commands.bot_has_guild_permissions(**perms)
    :decorator:

.. autofunction:: discord_real.ext.commands.bot_has_any_role(*items)
    :decorator:

.. autofunction:: discord_real.ext.commands.cooldown(rate, per, type=discord_real.ext.commands.BucketType.default)
    :decorator:

.. autofunction:: discord_real.ext.commands.dynamic_cooldown(cooldown, type)
    :decorator:

.. autofunction:: discord_real.ext.commands.max_concurrency(number, per=discord_real.ext.commands.BucketType.default, *, wait=False)
    :decorator:

.. autofunction:: discord_real.ext.commands.before_invoke(coro)
    :decorator:

.. autofunction:: discord_real.ext.commands.after_invoke(coro)
    :decorator:

.. autofunction:: discord_real.ext.commands.guild_only(,)
    :decorator:

.. autofunction:: discord_real.ext.commands.dm_only(,)
    :decorator:

.. autofunction:: discord_real.ext.commands.is_owner(,)
    :decorator:

.. autofunction:: discord_real.ext.commands.is_nsfw(,)
    :decorator:

.. _ext_commands_api_context:

Context
--------

.. attributetable:: discord_real.ext.commands.Context

.. autoclass:: discord_real.ext.commands.Context
    :members:
    :inherited-members:
    :exclude-members: typing

    .. automethod:: discord_real.ext.commands.Context.typing
        :async-with:

.. _ext_commands_api_converters:

Converters
------------

.. attributetable:: discord_real.ext.commands.Converter

.. autoclass:: discord_real.ext.commands.Converter
    :members:

.. attributetable:: discord_real.ext.commands.ObjectConverter

.. autoclass:: discord_real.ext.commands.ObjectConverter
    :members:

.. attributetable:: discord_real.ext.commands.MemberConverter

.. autoclass:: discord_real.ext.commands.MemberConverter
    :members:

.. attributetable:: discord_real.ext.commands.UserConverter

.. autoclass:: discord_real.ext.commands.UserConverter
    :members:

.. attributetable:: discord_real.ext.commands.MessageConverter

.. autoclass:: discord_real.ext.commands.MessageConverter
    :members:

.. attributetable:: discord_real.ext.commands.PartialMessageConverter

.. autoclass:: discord_real.ext.commands.PartialMessageConverter
    :members:

.. attributetable:: discord_real.ext.commands.GuildChannelConverter

.. autoclass:: discord_real.ext.commands.GuildChannelConverter
    :members:

.. attributetable:: discord_real.ext.commands.TextChannelConverter

.. autoclass:: discord_real.ext.commands.TextChannelConverter
    :members:

.. attributetable:: discord_real.ext.commands.VoiceChannelConverter

.. autoclass:: discord_real.ext.commands.VoiceChannelConverter
    :members:

.. attributetable:: discord_real.ext.commands.StageChannelConverter

.. autoclass:: discord_real.ext.commands.StageChannelConverter
    :members:

.. attributetable:: discord_real.ext.commands.CategoryChannelConverter

.. autoclass:: discord_real.ext.commands.CategoryChannelConverter
    :members:

.. attributetable:: discord_real.ext.commands.ForumChannelConverter

.. autoclass:: discord_real.ext.commands.ForumChannelConverter
    :members:

.. attributetable:: discord_real.ext.commands.InviteConverter

.. autoclass:: discord_real.ext.commands.InviteConverter
    :members:

.. attributetable:: discord_real.ext.commands.GuildConverter

.. autoclass:: discord_real.ext.commands.GuildConverter
    :members:

.. attributetable:: discord_real.ext.commands.RoleConverter

.. autoclass:: discord_real.ext.commands.RoleConverter
    :members:

.. attributetable:: discord_real.ext.commands.GameConverter

.. autoclass:: discord_real.ext.commands.GameConverter
    :members:

.. attributetable:: discord_real.ext.commands.ColourConverter

.. autoclass:: discord_real.ext.commands.ColourConverter
    :members:

.. attributetable:: discord_real.ext.commands.EmojiConverter

.. autoclass:: discord_real.ext.commands.EmojiConverter
    :members:

.. attributetable:: discord_real.ext.commands.PartialEmojiConverter

.. autoclass:: discord_real.ext.commands.PartialEmojiConverter
    :members:

.. attributetable:: discord_real.ext.commands.ThreadConverter

.. autoclass:: discord_real.ext.commands.ThreadConverter
    :members:

.. attributetable:: discord_real.ext.commands.GuildStickerConverter

.. autoclass:: discord_real.ext.commands.GuildStickerConverter
    :members:

.. attributetable:: discord_real.ext.commands.ScheduledEventConverter

.. autoclass:: discord_real.ext.commands.ScheduledEventConverter
    :members:

.. attributetable:: discord_real.ext.commands.clean_content

.. autoclass:: discord_real.ext.commands.clean_content
    :members:

.. attributetable:: discord_real.ext.commands.Greedy

.. autoclass:: discord_real.ext.commands.Greedy()

.. attributetable:: discord_real.ext.commands.Range

.. autoclass:: discord_real.ext.commands.Range()

.. autofunction:: discord_real.ext.commands.run_converters

Flag Converter
~~~~~~~~~~~~~~~

.. attributetable:: discord_real.ext.commands.FlagConverter

.. autoclass:: discord_real.ext.commands.FlagConverter
    :members:

.. attributetable:: discord_real.ext.commands.Flag

.. autoclass:: discord_real.ext.commands.Flag()
    :members:

.. autofunction:: discord_real.ext.commands.flag


Defaults
--------

.. attributetable:: discord_real.ext.commands.Parameter

.. autoclass:: discord_real.ext.commands.Parameter()
    :members:

.. autofunction:: discord_real.ext.commands.parameter

.. autofunction:: discord_real.ext.commands.param

.. data:: discord_real.ext.commands.Author

    A default :class:`Parameter` which returns the :attr:`~.Context.author` for this context.

    .. versionadded:: 2.0

.. data:: discord_real.ext.commands.CurrentChannel

    A default :class:`Parameter` which returns the :attr:`~.Context.channel` for this context.

    .. versionadded:: 2.0

.. data:: discord_real.ext.commands.CurrentGuild

    A default :class:`Parameter` which returns the :attr:`~.Context.guild` for this context. This will never be ``None``. If the command is called in a DM context then :exc:`~discord_real.ext.commands.NoPrivateMessage` is raised to the error handlers.

    .. versionadded:: 2.0

.. _ext_commands_api_errors:

Exceptions
-----------

.. autoexception:: discord_real.ext.commands.CommandError
    :members:

.. autoexception:: discord_real.ext.commands.ConversionError
    :members:

.. autoexception:: discord_real.ext.commands.MissingRequiredArgument
    :members:

.. autoexception:: discord_real.ext.commands.MissingRequiredAttachment
    :members:

.. autoexception:: discord_real.ext.commands.ArgumentParsingError
    :members:

.. autoexception:: discord_real.ext.commands.UnexpectedQuoteError
    :members:

.. autoexception:: discord_real.ext.commands.InvalidEndOfQuotedStringError
    :members:

.. autoexception:: discord_real.ext.commands.ExpectedClosingQuoteError
    :members:

.. autoexception:: discord_real.ext.commands.BadArgument
    :members:

.. autoexception:: discord_real.ext.commands.BadUnionArgument
    :members:

.. autoexception:: discord_real.ext.commands.BadLiteralArgument
    :members:

.. autoexception:: discord_real.ext.commands.PrivateMessageOnly
    :members:

.. autoexception:: discord_real.ext.commands.NoPrivateMessage
    :members:

.. autoexception:: discord_real.ext.commands.CheckFailure
    :members:

.. autoexception:: discord_real.ext.commands.CheckAnyFailure
    :members:

.. autoexception:: discord_real.ext.commands.CommandNotFound
    :members:

.. autoexception:: discord_real.ext.commands.DisabledCommand
    :members:

.. autoexception:: discord_real.ext.commands.CommandInvokeError
    :members:

.. autoexception:: discord_real.ext.commands.TooManyArguments
    :members:

.. autoexception:: discord_real.ext.commands.UserInputError
    :members:

.. autoexception:: discord_real.ext.commands.CommandOnCooldown
    :members:

.. autoexception:: discord_real.ext.commands.MaxConcurrencyReached
    :members:

.. autoexception:: discord_real.ext.commands.NotOwner
    :members:

.. autoexception:: discord_real.ext.commands.MessageNotFound
    :members:

.. autoexception:: discord_real.ext.commands.MemberNotFound
    :members:

.. autoexception:: discord_real.ext.commands.GuildNotFound
    :members:

.. autoexception:: discord_real.ext.commands.UserNotFound
    :members:

.. autoexception:: discord_real.ext.commands.ChannelNotFound
    :members:

.. autoexception:: discord_real.ext.commands.ChannelNotReadable
    :members:

.. autoexception:: discord_real.ext.commands.ThreadNotFound
    :members:

.. autoexception:: discord_real.ext.commands.BadColourArgument
    :members:

.. autoexception:: discord_real.ext.commands.RoleNotFound
    :members:

.. autoexception:: discord_real.ext.commands.BadInviteArgument
    :members:

.. autoexception:: discord_real.ext.commands.EmojiNotFound
    :members:

.. autoexception:: discord_real.ext.commands.PartialEmojiConversionFailure
    :members:

.. autoexception:: discord_real.ext.commands.GuildStickerNotFound
    :members:

.. autoexception:: discord_real.ext.commands.ScheduledEventNotFound
    :members:

.. autoexception:: discord_real.ext.commands.BadBoolArgument
    :members:

.. autoexception:: discord_real.ext.commands.RangeError
    :members:

.. autoexception:: discord_real.ext.commands.MissingPermissions
    :members:

.. autoexception:: discord_real.ext.commands.BotMissingPermissions
    :members:

.. autoexception:: discord_real.ext.commands.MissingRole
    :members:

.. autoexception:: discord_real.ext.commands.BotMissingRole
    :members:

.. autoexception:: discord_real.ext.commands.MissingAnyRole
    :members:

.. autoexception:: discord_real.ext.commands.BotMissingAnyRole
    :members:

.. autoexception:: discord_real.ext.commands.NSFWChannelRequired
    :members:

.. autoexception:: discord_real.ext.commands.FlagError
    :members:

.. autoexception:: discord_real.ext.commands.BadFlagArgument
    :members:

.. autoexception:: discord_real.ext.commands.MissingFlagArgument
    :members:

.. autoexception:: discord_real.ext.commands.TooManyFlags
    :members:

.. autoexception:: discord_real.ext.commands.MissingRequiredFlag
    :members:

.. autoexception:: discord_real.ext.commands.ExtensionError
    :members:

.. autoexception:: discord_real.ext.commands.ExtensionAlreadyLoaded
    :members:

.. autoexception:: discord_real.ext.commands.ExtensionNotLoaded
    :members:

.. autoexception:: discord_real.ext.commands.NoEntryPointError
    :members:

.. autoexception:: discord_real.ext.commands.ExtensionFailed
    :members:

.. autoexception:: discord_real.ext.commands.ExtensionNotFound
    :members:

.. autoexception:: discord_real.ext.commands.CommandRegistrationError
    :members:

.. autoexception:: discord_real.ext.commands.HybridCommandError
    :members:


Exception Hierarchy
~~~~~~~~~~~~~~~~~~~~~

.. exception_hierarchy::

    - :exc:`~.DiscordException`
        - :exc:`~.commands.CommandError`
            - :exc:`~.commands.ConversionError`
            - :exc:`~.commands.UserInputError`
                - :exc:`~.commands.MissingRequiredArgument`
                - :exc:`~.commands.MissingRequiredAttachment`
                - :exc:`~.commands.TooManyArguments`
                - :exc:`~.commands.BadArgument`
                    - :exc:`~.commands.MessageNotFound`
                    - :exc:`~.commands.MemberNotFound`
                    - :exc:`~.commands.GuildNotFound`
                    - :exc:`~.commands.UserNotFound`
                    - :exc:`~.commands.ChannelNotFound`
                    - :exc:`~.commands.ChannelNotReadable`
                    - :exc:`~.commands.BadColourArgument`
                    - :exc:`~.commands.RoleNotFound`
                    - :exc:`~.commands.BadInviteArgument`
                    - :exc:`~.commands.EmojiNotFound`
                    - :exc:`~.commands.GuildStickerNotFound`
                    - :exc:`~.commands.ScheduledEventNotFound`
                    - :exc:`~.commands.PartialEmojiConversionFailure`
                    - :exc:`~.commands.BadBoolArgument`
                    - :exc:`~.commands.RangeError`
                    - :exc:`~.commands.ThreadNotFound`
                    - :exc:`~.commands.FlagError`
                        - :exc:`~.commands.BadFlagArgument`
                        - :exc:`~.commands.MissingFlagArgument`
                        - :exc:`~.commands.TooManyFlags`
                        - :exc:`~.commands.MissingRequiredFlag`
                - :exc:`~.commands.BadUnionArgument`
                - :exc:`~.commands.BadLiteralArgument`
                - :exc:`~.commands.ArgumentParsingError`
                    - :exc:`~.commands.UnexpectedQuoteError`
                    - :exc:`~.commands.InvalidEndOfQuotedStringError`
                    - :exc:`~.commands.ExpectedClosingQuoteError`
            - :exc:`~.commands.CommandNotFound`
            - :exc:`~.commands.CheckFailure`
                - :exc:`~.commands.CheckAnyFailure`
                - :exc:`~.commands.PrivateMessageOnly`
                - :exc:`~.commands.NoPrivateMessage`
                - :exc:`~.commands.NotOwner`
                - :exc:`~.commands.MissingPermissions`
                - :exc:`~.commands.BotMissingPermissions`
                - :exc:`~.commands.MissingRole`
                - :exc:`~.commands.BotMissingRole`
                - :exc:`~.commands.MissingAnyRole`
                - :exc:`~.commands.BotMissingAnyRole`
                - :exc:`~.commands.NSFWChannelRequired`
            - :exc:`~.commands.DisabledCommand`
            - :exc:`~.commands.CommandInvokeError`
            - :exc:`~.commands.CommandOnCooldown`
            - :exc:`~.commands.MaxConcurrencyReached`
            - :exc:`~.commands.HybridCommandError`
        - :exc:`~.commands.ExtensionError`
            - :exc:`~.commands.ExtensionAlreadyLoaded`
            - :exc:`~.commands.ExtensionNotLoaded`
            - :exc:`~.commands.NoEntryPointError`
            - :exc:`~.commands.ExtensionFailed`
            - :exc:`~.commands.ExtensionNotFound`
    - :exc:`~.ClientException`
        - :exc:`~.commands.CommandRegistrationError`
