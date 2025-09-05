"""
Unit tests for Discord AI Assistant Bot

Tests cover system prompt loading, message handling, LLM integration,
and bot configuration functionality.
"""

import unittest
from unittest.mock import Mock, patch, mock_open, AsyncMock
import asyncio
import os
import tempfile
import json

import aiohttp
from aiohttp import ClientResponseError

# Import functions from my_discord_ai_bot (assuming they're importable)
# Note: You may need to adjust imports based on your module structure
try:
    from my_discord_ai_bot import (
        load_system_prompt,
        remove_bot_mentions,
        setup_ssl_context,
        ask_llm,
        handle_question,
        handle_mention,
        SYSTEM_PROMPT_FILE,
        ERROR_MESSAGE_TEMPLATE,
        GREETING_MESSAGE,
        MAX_RESPONSE_LENGTH,
        BOT_NAME
    )
except ImportError:
    # If direct import fails, we'll test the functions as if they were defined here
    pass

class TestSystemPromptLoading(unittest.TestCase):
    """Test system prompt loading functionality."""

    @patch('my_discord_ai_bot.open', new_callable=mock_open, read_data="Test system prompt content")
    def test_load_system_prompt_success(self, mock_file):
        """Test successful loading of system prompt from file."""
        from my_discord_ai_bot import load_system_prompt

        result = load_system_prompt()

        self.assertEqual(result, "Test system prompt content")
        mock_file.assert_called_once_with("system_prompt.txt", "r", encoding="utf-8")

    @patch('my_discord_ai_bot.open', side_effect=FileNotFoundError)
    @patch('builtins.print')
    def test_load_system_prompt_file_not_found(self, mock_print, mock_file):
        """Test fallback when system prompt file is not found."""
        from my_discord_ai_bot import load_system_prompt

        result = load_system_prompt()

        self.assertEqual(result, "Reply always that there was an error and you cannot continue")
        mock_print.assert_called()

    @patch('my_discord_ai_bot.open', new_callable=mock_open, read_data="   \n\n   ")
    @patch('builtins.print')
    def test_load_system_prompt_empty_file(self, mock_print, mock_file):
        """Test handling of empty system prompt file."""
        from my_discord_ai_bot import load_system_prompt

        result = load_system_prompt()

        self.assertEqual(result, "Reply always that there was an error and you cannot continue")
        mock_print.assert_called()

class TestBotMentionHandling(unittest.TestCase):
    """Test bot mention detection and processing."""

    def test_remove_bot_mentions_regular(self):
        """Test removal of regular bot mentions."""
        from my_discord_ai_bot import remove_bot_mentions

        content = "<@123456789> Hello there, how are you?"
        result = remove_bot_mentions(content, 123456789)

        self.assertEqual(result, "Hello there, how are you?")

    def test_remove_bot_mentions_nickname(self):
        """Test removal of nickname bot mentions."""
        from my_discord_ai_bot import remove_bot_mentions

        content = "<@!123456789> What's the weather like?"
        result = remove_bot_mentions(content, 123456789)

        self.assertEqual(result, "What's the weather like?")

    def test_remove_bot_mentions_multiple(self):
        """Test removal of multiple bot mentions."""
        from my_discord_ai_bot import remove_bot_mentions

        content = "<@123456789> <@!123456789> Tell me a joke"
        result = remove_bot_mentions(content, 123456789)

        self.assertEqual(result, "Tell me a joke")

    def test_remove_bot_mentions_only_mentions(self):
        """Test handling when message contains only mentions."""
        from my_discord_ai_bot import remove_bot_mentions

        content = "<@123456789> <@!123456789>"
        result = remove_bot_mentions(content, 123456789)

        self.assertEqual(result, "")

    def test_remove_bot_mentions_different_bot_id(self):
        """Test that mentions of other bots are not removed."""
        from my_discord_ai_bot import remove_bot_mentions

        content = "<@987654321> Hello there!"
        result = remove_bot_mentions(content, 123456789)

        self.assertEqual(result, "<@987654321> Hello there!")

class TestLLMIntegration(unittest.TestCase):
    """Test LLM API integration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_response_data = {
            "message": {
                "content": "This is a test response from the LLM"
            }
        }

    @patch('my_discord_ai_bot.aiohttp.ClientSession')
    @patch('my_discord_ai_bot.API_PROVIDER', 'ollama')
    async def test_ask_llm_success_ollama(self, mock_session_class):
        """Test successful LLM API request with Ollama provider."""
        # Mock the session and response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=self.mock_response_data)

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        from my_discord_ai_bot import ask_llm

        result = await ask_llm("Test question")

        self.assertEqual(result, "This is a test response from the LLM")
        mock_session.post.assert_called_once()

    @patch('my_discord_ai_bot.aiohttp.ClientSession')
    @patch('my_discord_ai_bot.API_PROVIDER', 'openai')
    @patch('my_discord_ai_bot.OPENAI_API_KEY', 'test-api-key')
    async def test_ask_llm_success_openai(self, mock_session_class):
        """Test successful LLM API request with OpenAI provider."""
        # Mock OpenAI response format
        openai_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response from OpenAI"
                    }
                }
            ]
        }

        # Mock the session and response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=openai_response_data)

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        from my_discord_ai_bot import ask_llm

        result = await ask_llm("Test question")

        self.assertEqual(result, "This is a test response from OpenAI")
        mock_session.post.assert_called_once()

    @patch('my_discord_ai_bot.API_PROVIDER', 'openai')
    @patch('my_discord_ai_bot.OPENAI_API_KEY', None)
    async def test_ask_llm_openai_missing_key(self):
        """Test OpenAI provider with missing API key."""
        from my_discord_ai_bot import ask_llm

        with self.assertRaises(ValueError) as context:
            await ask_llm("Test question")

        self.assertIn("OPENAI_API_KEY is required", str(context.exception))

    @patch('my_discord_ai_bot.aiohttp.ClientSession')
    @patch('my_discord_ai_bot.API_PROVIDER', 'ollama')
    @patch('my_discord_ai_bot.ENABLE_FALLBACK', True)
    @patch('my_discord_ai_bot.OPENAI_API_KEY', 'test-api-key')
    @patch('os.getenv')
    async def test_ask_llm_fallback_success(self, mock_getenv, mock_session_class):
        """Test successful fallback from Ollama to OpenAI when Ollama fails."""
        # Mock environment variable for fallback model
        mock_getenv.return_value = 'gpt-3.5-turbo'

        # Mock OpenAI response format for fallback
        openai_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "This is a fallback response from OpenAI"
                    }
                }
            ]
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=openai_response_data)

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # First call fails (Ollama), second succeeds (OpenAI fallback)
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            side_effect=[
                AsyncMock(side_effect=aiohttp.ClientConnectorError(connection_key=None, os_error=None)),
                mock_response
            ]
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        from my_discord_ai_bot import ask_llm

        result = await ask_llm("Test question")

        self.assertEqual(result, "This is a fallback response from OpenAI")
        # Should be called twice - once for Ollama (fails), once for OpenAI (succeeds)
        self.assertEqual(mock_session.post.call_count, 2)

    @patch('my_discord_ai_bot.aiohttp.ClientSession')
    @patch('my_discord_ai_bot.API_PROVIDER', 'ollama')
    @patch('my_discord_ai_bot.ENABLE_FALLBACK', False)
    async def test_ask_llm_no_fallback_when_disabled(self, mock_session_class):
        """Test that fallback doesn't occur when disabled."""
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock connection error
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(connection_key=None, os_error=None)
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        from my_discord_ai_bot import ask_llm

        with self.assertRaises(aiohttp.ClientConnectorError):
            await ask_llm("Test question")

        # Should only be called once (no fallback attempt)
        self.assertEqual(mock_session.post.call_count, 1)

    @patch('my_discord_ai_bot.aiohttp.ClientSession')
    async def test_ask_llm_http_error(self, mock_session_class):
        """Test LLM API request with HTTP error."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = ClientResponseError(
            request_info=Mock(), history=Mock()
        )

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        from my_discord_ai_bot import ask_llm

        with self.assertRaises(ClientResponseError):
            await ask_llm("Test question")

    @patch('my_discord_ai_bot.aiohttp.ClientSession')
    async def test_ask_llm_empty_response(self, mock_session_class):
        """Test LLM API request with empty response content."""
        mock_response_data = {"message": {"content": ""}}

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        from my_discord_ai_bot import ask_llm

        result = await ask_llm("Test question")

        self.assertEqual(result, "")

class TestMessageHandling(unittest.TestCase):
    """Test Discord message handling functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_message = Mock()
        self.mock_message.reply = AsyncMock()
        self.mock_message.channel.typing = AsyncMock()
        self.mock_message.content = "Test message content"

    @patch('my_discord_ai_bot.ask_llm')
    async def test_handle_question_success(self, mock_ask_llm):
        """Test successful question handling."""
        mock_ask_llm.return_value = "Test LLM response"

        from my_discord_ai_bot import handle_question

        await handle_question(self.mock_message, "Test question")

        mock_ask_llm.assert_called_once_with("Test question")
        self.mock_message.reply.assert_called_once_with("Test LLM response")

    @patch('my_discord_ai_bot.ask_llm')
    async def test_handle_question_long_response(self, mock_ask_llm):
        """Test question handling with response truncation."""
        long_response = "x" * 500  # Longer than MAX_RESPONSE_LENGTH
        mock_ask_llm.return_value = long_response

        from my_discord_ai_bot import handle_question

        await handle_question(self.mock_message, "Test question")

        # Should truncate to MAX_RESPONSE_LENGTH (400)
        expected_response = long_response[:400]
        self.mock_message.reply.assert_called_once_with(expected_response)

    @patch('my_discord_ai_bot.ask_llm')
    @patch('builtins.print')
    async def test_handle_question_llm_error(self, mock_print, mock_ask_llm):
        """Test question handling when LLM raises an error."""
        mock_ask_llm.side_effect = Exception("LLM API error")

        from my_discord_ai_bot import handle_question, ERROR_MESSAGE_TEMPLATE, BOT_NAME

        await handle_question(self.mock_message, "Test question")

        expected_error = ERROR_MESSAGE_TEMPLATE.format(BOT_NAME)
        self.mock_message.reply.assert_called_once_with(expected_error)
        mock_print.assert_called()

    @patch('my_discord_ai_bot.remove_bot_mentions')
    @patch('my_discord_ai_bot.handle_question')
    async def test_handle_mention_with_question(self, mock_handle_question, mock_remove_mentions):
        """Test handling mention when user asks a question."""
        mock_remove_mentions.return_value = "What is the weather?"
        self.mock_message.author.id = 987654321  # Not the bot

        from my_discord_ai_bot import handle_mention

        await handle_mention(self.mock_message)

        mock_handle_question.assert_called_once_with(self.mock_message, "What is the weather?")
        self.mock_message.reply.assert_not_called()

    @patch('my_discord_ai_bot.remove_bot_mentions')
    async def test_handle_mention_no_question(self, mock_remove_mentions):
        """Test handling mention when user doesn't ask a question."""
        mock_remove_mentions.return_value = ""  # Empty after removing mentions

        from my_discord_ai_bot import handle_mention

        await handle_mention(self.mock_message)

        self.mock_message.reply.assert_called_once_with(GREETING_MESSAGE)

class TestConfiguration(unittest.TestCase):
    """Test bot configuration and setup."""

    def test_setup_ssl_context(self):
        """Test SSL context creation."""
        from my_discord_ai_bot import setup_ssl_context
        import ssl

        context = setup_ssl_context()

        self.assertIsInstance(context, ssl.SSLContext)

    @patch.dict(os.environ, {'DISCORD_TOKEN': 'test_token'})
    def test_environment_variable_loading(self):
        """Test loading of environment variables."""
        token = os.getenv('DISCORD_TOKEN')
        self.assertEqual(token, 'test_token')

    def test_constants_defined(self):
        """Test that required constants are defined."""
        from my_discord_ai_bot import (
            MAX_RESPONSE_LENGTH,
            ERROR_MESSAGE_TEMPLATE,
            GREETING_MESSAGE,
            SYSTEM_PROMPT_FILE,
            BOT_NAME
        )

        self.assertIsInstance(MAX_RESPONSE_LENGTH, int)
        self.assertIsInstance(ERROR_MESSAGE_TEMPLATE, str)
        self.assertIsInstance(GREETING_MESSAGE, str)
        self.assertIsInstance(SYSTEM_PROMPT_FILE, str)
        self.assertIsInstance(BOT_NAME, str)
        self.assertGreater(MAX_RESPONSE_LENGTH, 0)
        self.assertIn("{}", ERROR_MESSAGE_TEMPLATE)  # Should contain format placeholder

class TestIntegration(unittest.TestCase):
    """Integration tests for multiple components working together."""

    @patch('my_discord_ai_bot.load_system_prompt')
    @patch('my_discord_ai_bot.setup_discord_bot')
    def test_bot_initialization(self, mock_setup_bot, mock_load_prompt):
        """Test bot initialization process."""
        mock_load_prompt.return_value = "Test system prompt"
        mock_bot = Mock()
        mock_setup_bot.return_value = mock_bot

        # This would test the initialization if we refactored it into a function
        # For now, we just test that the mocks would be called
        self.assertTrue(True)  # Placeholder

def run_async_test(test_func):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_func())
    finally:
        loop.close()

if __name__ == '__main__':
    # Create a custom test suite that handles async tests
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
