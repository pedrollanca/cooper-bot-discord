"""
Discord AI Assistant for a Discord Community

A Discord bot that uses Ollama/LLM API to provide AI-powered responses
when mentioned in chat.
"""

import os
import aiohttp
import asyncio
import discord
import ssl
import certifi
from discord.ext import commands
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# File paths
# Check system_prompt_demo.txt for a sample prompt file
SYSTEM_PROMPT_FILE = "system_prompt.txt"
ENV_FILE = ".env"

# Bot configuration
COMMAND_PREFIX = "!"
MAX_RESPONSE_LENGTH = 400
TYPING_TIMEOUT = 30  # seconds

# LLM configuration defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 120

# Error messages (will be formatted with bot name)
ERROR_MESSAGE_TEMPLATE = "⚠️ {} had a hiccup, try again!"
GREETING_MESSAGE = "Hey there! 🤖 What can I help you with?"
STARTUP_MESSAGE_TEMPLATE = "✅ {} logged in as {}"

# =============================================================================
# INITIALIZATION
# =============================================================================

def setup_ssl_context():
    """Create SSL context with proper certificates for secure connections."""
    return ssl.create_default_context(cafile=certifi.where())

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv(ENV_FILE)

def load_system_prompt():
    """
    Load system prompt from file with fallback error message.

    Returns:
        str: The system prompt content or error fallback
    """
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
            if not prompt:
                raise ValueError("System prompt file is empty")
            return prompt
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load system prompt - {e}")
        return "Reply always that there was an error and you cannot continue"

def setup_discord_bot():
    """
    Initialize Discord bot with proper intents and configuration.

    Returns:
        commands.Bot: Configured Discord bot instance
    """
    intents = discord.Intents.default()
    intents.message_content = True
    return commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

# Initialize components
ssl_context = setup_ssl_context()
load_environment_variables()

# Load configuration from environment
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA_URL = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
MODEL = os.getenv("MODEL", DEFAULT_MODEL)
TEMPERATURE = float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS))
BOT_NAME = os.getenv("BOT_NAME", "cooperbot")

# Load system prompt and initialize bot
SYSTEM_PROMPT = load_system_prompt()
bot = setup_discord_bot()

# =============================================================================
# LLM INTEGRATION
# =============================================================================

async def ask_llm(user_input: str) -> str:
    """
    Send user input to LLM API and return the response.

    Args:
        user_input (str): The user's question or message

    Returns:
        str: The LLM's response

    Raises:
        aiohttp.ClientError: If API request fails
        KeyError: If response format is unexpected
    """
    # Create secure connector with custom SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Prepare API payload
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS
            },
            "stream": False,
        }

        # Make API request
        async with session.post(OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

            # Extract and return response content
            return data.get("message", {}).get("content", "").strip()

# =============================================================================
# DISCORD EVENT HANDLERS
# =============================================================================

@bot.event
async def on_ready():
    """Handle bot startup and log successful connection."""
    print(STARTUP_MESSAGE_TEMPLATE.format(BOT_NAME, bot.user))

@bot.event
async def on_message(message):
    """
    Handle incoming Discord messages and respond to mentions.

    Args:
        message (discord.Message): The incoming message
    """
    # Ignore bot's own messages to prevent loops
    if message.author == bot.user:
        return

    # Check if bot is mentioned in the message
    if bot.user in message.mentions:
        await handle_mention(message)

    # Process other commands (for future slash commands or other functionality)
    await bot.process_commands(message)

async def handle_mention(message):
    """
    Handle when the bot is mentioned in a message.

    Args:
        message (discord.Message): The message that mentioned the bot
    """
    # Extract question by removing bot mentions
    question = remove_bot_mentions(message.content, bot.user.id)

    if question:
        # User asked a question
        await handle_question(message, question)
    else:
        # User just mentioned the bot without a question
        await message.reply(GREETING_MESSAGE)

def remove_bot_mentions(content: str, bot_id: int) -> str:
    """
    Remove bot mentions from message content.

    Args:
        content (str): Original message content
        bot_id (int): Bot's user ID

    Returns:
        str: Message content with bot mentions removed
    """
    # Remove both regular and nickname mentions
    content = content.replace(f'<@{bot_id}>', '')
    content = content.replace(f'<@!{bot_id}>', '')
    return content.strip()

async def handle_question(message, question: str):
    """
    Process a user's question and respond with LLM output.

    Args:
        message (discord.Message): The original message
        question (str): The extracted question content
    """
    async with message.channel.typing():
        try:
            # Get response from LLM
            reply = await ask_llm(question)

            # Truncate response if too long and send reply
            truncated_reply = reply[:MAX_RESPONSE_LENGTH]
            await message.reply(truncated_reply)

        except Exception as e:
            # Log error and send user-friendly error message
            print(f"Error processing question: {e}")
            await message.reply(ERROR_MESSAGE_TEMPLATE.format(BOT_NAME))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to start the Discord bot."""
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables")
        return

    # Start the bot
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()