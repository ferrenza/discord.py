# This example requires the 'message_content' privileged intent to function.

from discord_real.ext import commands

import discord_real
from urllib.parse import quote_plus


class GoogleBot(commands.Bot):
    def __init__(self):
        intents = discord_real.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=commands.when_mentioned_or('$'), intents=intents)

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')


# Define a simple View that gives us a google link button.
# We take in `query` as the query that the command author requests for
class Google(discord_real.ui.View):
    def __init__(self, query: str):
        super().__init__()
        # we need to quote the query string to make a valid url. Discord will raise an error if it isn't valid.
        query = quote_plus(query)
        url = f'https://www.google.com/search?q={query}'

        # Link buttons cannot be made with the decorator
        # Therefore we have to manually create one.
        # We add the quoted url to the button, and add the button to the view.
        self.add_item(discord_real.ui.Button(label='Click Here', url=url))


bot = GoogleBot()


@bot.command()
async def google(ctx: commands.Context, *, query: str):
    """Returns a google link for a query"""
    await ctx.send(f'Google Result for: `{query}`', view=Google(query))


bot.run('token')
