"""
Discord AI Assistant for a Discord Community

A Discord bot that uses local LLM API (Ollama) or OpenAI ChatGPT API 
to provide AI-powered responses when mentioned in chat.
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
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 120

# API provider types
API_PROVIDER_OLLAMA = "ollama"
API_PROVIDER_OPENAI = "openai"
API_PROVIDER_FALLBACK = "fallback"

# Timeout settings (in seconds)
LOCAL_LLM_TIMEOUT = 5   # Short timeout for local LLM since it should be fast
REMOTE_API_TIMEOUT = 30 # Longer timeout for remote APIs like OpenAI

# Error messages (will be formatted with bot name)
ERROR_MESSAGE_TEMPLATE = "⚠️ {} had a hiccup, try again!"
GREETING_MESSAGE = "Hey there! 🤖 What can I help you with?"
STARTUP_MESSAGE_TEMPLATE = "✅ {} logged in as {}"
FALLBACK_MESSAGE_TEMPLATE = "⚠️ Local LLM unavailable, using OpenAI fallback"

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
API_PROVIDER = os.getenv("API_PROVIDER", API_PROVIDER_FALLBACK).lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL", DEFAULT_OPENAI_URL)

# Fallback configuration
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "true").lower() in ("true", "1", "yes")

# Set default models for each provider
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

# Set primary model based on provider
if API_PROVIDER == API_PROVIDER_OPENAI:
    MODEL = OPENAI_MODEL
    API_URL = OPENAI_URL
elif API_PROVIDER == API_PROVIDER_OLLAMA:
    MODEL = OLLAMA_MODEL
    API_URL = OLLAMA_URL
else:  # fallback mode
    MODEL = OLLAMA_MODEL  # Try Ollama first
    API_URL = OLLAMA_URL

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
    Supports both local LLM (Ollama) and OpenAI ChatGPT APIs.
    When using Ollama as primary, falls back to OpenAI if Ollama is unavailable.

    Args:
        user_input (str): The user's question or message

    Returns:
        str: The LLM's response

    Raises:
        aiohttp.ClientError: If API request fails on all providers
        KeyError: If response format is unexpected
        ValueError: If OpenAI API key is missing when needed
    """
    # Create secure connector with custom SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Try primary provider first
        try:
            return await _make_api_request(session, user_input, API_PROVIDER, API_URL, MODEL)
        except (aiohttp.ClientError, asyncio.TimeoutError) as primary_error:
            # If primary provider is Ollama and fails, try OpenAI fallback
            if API_PROVIDER == API_PROVIDER_OLLAMA and ENABLE_FALLBACK and OPENAI_API_KEY:
                print(f"Warning: Primary LLM API failed, trying OpenAI fallback...")
                try:
                    fallback_model = os.getenv("FALLBACK_MODEL", DEFAULT_OPENAI_MODEL)
                    return await _make_api_request(session, user_input, API_PROVIDER_OPENAI, 
                                                 OPENAI_URL, fallback_model)
                except Exception as fallback_error:
                    print(f"Error: Fallback API also failed ({fallback_error})")
                    raise primary_error
            else:
                # Re-raise original error if no fallback available
                raise primary_error


async def _make_api_request(session: aiohttp.ClientSession, user_input: str, 
                          provider: str, api_url: str, model: str) -> str:
    """
    Make an API request to the specified LLM provider.

    Args:
        session: aiohttp session
        user_input: user's message
        provider: API provider type ("ollama" or "openai")
        api_url: API endpoint URL
        model: model identifier

    Returns:
        str: The LLM's response content
    """
    # Prepare headers and payload based on API provider
    headers = {"Content-Type": "application/json"}

    if provider == API_PROVIDER_OPENAI:
        # OpenAI API configuration
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")

        # Ensure proper authorization header format
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY.strip()}"

        # Ensure temperature is within valid range (0.0 to 2.0)
        openai_temperature = max(0.0, min(2.0, float(TEMPERATURE)))

        # Ensure max_tokens is positive and reasonable (OpenAI has model-specific limits)
        openai_max_tokens = max(1, min(4096, int(MAX_TOKENS)))

        # Build messages array - ensure content is string and not empty
        messages = []
        system_content = str(SYSTEM_PROMPT).strip() if SYSTEM_PROMPT else ""
        if system_content:
            messages.append({"role": "system", "content": system_content})
            print(f"Debug: System prompt length: {len(system_content)}")
        else:
            print("Debug: No system prompt provided")

        user_content = str(user_input).strip()
        if not user_content:
            raise ValueError("User input cannot be empty")
        messages.append({"role": "user", "content": user_content})
        print(f"Debug: User content: '{user_content}' (length: {len(user_content)})")

        # Validate model name for OpenAI
        model_name = str(model).strip()
        if not model_name:
            model_name = "gpt-3.5-turbo"  # Default fallback

        # Build payload according to current OpenAI API spec
        payload = {
            "model": model_name,
            "messages": messages,
            # "max_completion_tokens": openai_max_tokens,
        }

        # Validate payload
        if not payload.get("messages") or len(payload["messages"]) == 0:
            raise ValueError("Messages cannot be empty for OpenAI API")

    else:
        # Ollama/Local LLM API configuration
        payload = {
            "model": model,
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

    # Set timeout based on provider type
    timeout_duration = LOCAL_LLM_TIMEOUT if provider == API_PROVIDER_OLLAMA else REMOTE_API_TIMEOUT
    timeout = aiohttp.ClientTimeout(total=timeout_duration)

    async with session.post(api_url, json=payload, headers=headers, timeout=timeout) as response:
        if not response.ok:
            # Get the error response for debugging
            error_text = await response.text()
            print(f"API Error ({response.status}): {error_text}")
            response.raise_for_status()

        data = await response.json()

        # Extract response content based on API provider
        if provider == API_PROVIDER_OPENAI:
            response_content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print(f"Debug: OpenAI response content: '{response_content}' (length: {len(response_content)})")
            return response_content
        else:
            response_content = data.get("message", {}).get("content", "").strip()
            print(f"Debug: Ollama response content: '{response_content}' (length: {len(response_content)})")
            return response_content

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
            print(f"Debug: Raw LLM reply: '{reply}' (length: {len(reply)})")

            # Ensure we have a valid response
            if not reply or not reply.strip():
                print("Warning: LLM returned empty response")
                reply = "I'm sorry, I couldn't generate a response right now."

            # Truncate response if too long and send reply
            truncated_reply = reply[:MAX_RESPONSE_LENGTH]
            if truncated_reply.strip():  # Final check before sending
                await message.reply(truncated_reply)
            else:
                await message.reply("I'm having trouble responding right now. Please try again!")

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

    # Validate configuration based on API provider
    if API_PROVIDER == API_PROVIDER_OPENAI and not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is required when API_PROVIDER is set to 'openai'")
        return

    print(f"Starting bot with {API_PROVIDER.upper()} API provider...")
    print(f"Model: {MODEL}")

    # Show fallback status
    if API_PROVIDER == API_PROVIDER_OLLAMA and ENABLE_FALLBACK:
        if OPENAI_API_KEY:
            fallback_model = os.getenv("FALLBACK_MODEL", DEFAULT_OPENAI_MODEL)
            print(f"OpenAI fallback enabled (model: {fallback_model})")
        else:
            print("Warning: Fallback enabled but OPENAI_API_KEY not set")

    # Start the bot
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()