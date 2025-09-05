# Discord AI Assistant Bot

A Discord bot that replies to mentions with AI-generated text using either local LLM APIs (like Ollama) or OpenAI's ChatGPT API. The bot's behavior is controlled by a plain-text system prompt and runtime configuration via environment variables.

Purpose
-------
Provide a flexible AI assistant that can be deployed to Discord servers with support for both local and cloud-based language models. The design prioritizes simplicity, testability, and easy switching between API providers.

Technical summary
-----------------
- Language: Python (asyncio)
- Discord integration: discord.py
- HTTP client: aiohttp (async)
- API support: Local LLM (Ollama-compatible) and OpenAI ChatGPT
- Configuration: environment variables + system prompt file
- SSL: system certificates via certifi
- Tests: unittest with async support; test runner included

Repository layout
-----------------
- my_discord_ai_bot.py     — main bot implementation and event handlers
- system_prompt.txt        — system prompt that shapes the bot's responses
- system_prompt_demo.txt   — example prompt template
- .env.example             — environment variable template
- requirements.txt         — Python dependencies
- test_my_discord_ai_bot.py— unit tests covering core functionality
- test_runner.py           — async-aware test runner

Quick start
-----------
1. Create and activate a Python virtual environment:
   - Linux / macOS:
     python -m venv venv
     source venv/bin/activate
   - Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

2. Install dependencies:
   pip install -r requirements.txt

3. Configure environment:
   Copy `.env.example` to `.env` and set the required values:
   - DISCORD_TOKEN: Discord bot token (required)
   - API_PROVIDER: set to "ollama" for local LLM or "openai" for OpenAI ChatGPT
   - For local LLM: OLLAMA_URL, MODEL (e.g., llama3.1:8b)
   - For OpenAI: OPENAI_API_KEY, MODEL (e.g., gpt-3.5-turbo or gpt-4)
   - TEMPERATURE, MAX_TOKENS: model parameters

4. Edit system prompt:
   Update `system_prompt.txt` to define the bot's personality, safety constraints, and reply style.

5. Start the bot:
   python my_discord_ai_bot.py

How it works
------------
- The bot listens for messages and only acts when it is mentioned.
- On mention, the bot removes mention tokens, prepares a payload with the system prompt and user content, and sends it to the configured API (local LLM or OpenAI).
- If using Ollama as the primary provider and it fails (e.g., not running), the bot can automatically fallback to OpenAI.
- The API response is trimmed to a configurable maximum length and sent back as a reply.
- Secure HTTP connections use the system certificate bundle via certifi.

Configuration details
---------------------
Key environment variables (from `.env.example`):
- DISCORD_TOKEN — required bot token
- API_PROVIDER — "ollama" (default) or "openai"
- BOT_NAME — optional display name in logs

For local LLM (API_PROVIDER=ollama):
- OLLAMA_URL — LLM API endpoint (defaults to localhost Ollama)
- MODEL — model identifier (e.g., llama3.1:8b, codellama:13b)

For OpenAI (API_PROVIDER=openai or fallback):
- OPENAI_API_KEY — your OpenAI API key (required for OpenAI usage)
- OPENAI_URL — OpenAI API endpoint (defaults to official endpoint)
- MODEL — OpenAI model (e.g., gpt-3.5-turbo, gpt-4, gpt-4-turbo)

Fallback configuration (when API_PROVIDER=ollama):
- ENABLE_FALLBACK — "true" (default) to enable OpenAI fallback when Ollama fails
- FALLBACK_MODEL — OpenAI model to use for fallback (defaults to gpt-3.5-turbo)

Common parameters:
- TEMPERATURE — sampling temperature for generation (0.0-2.0)
- MAX_TOKENS — max tokens for the request

Testing
-------
- Run the provided async-aware test runner:
  python test_runner.py

- Tests cover:
  - system prompt loading and fallback
  - mention parsing and message processing
  - LLM request/response handling (using mocked aiohttp)
  - configuration and SSL context validation

Troubleshooting
---------------
- Bot fails to start: ensure `DISCORD_TOKEN` is present in `.env` and valid.
- For local LLM: verify `OLLAMA_URL` is reachable and the specified model is available. If Ollama is down, the bot will attempt to fallback to OpenAI if configured.
- For OpenAI: ensure `OPENAI_API_KEY` is valid and has sufficient credits/quota.
- Fallback not working: verify `ENABLE_FALLBACK=true` and `OPENAI_API_KEY` is set when using Ollama as primary.
- API errors: check that the specified model exists and is accessible via your chosen provider.
- System prompt missing/empty: `system_prompt.txt` must contain the instructions that shape responses; a fallback message exists but is not recommended for production.
- Dependency issues: confirm the virtual environment is active and `pip install -r requirements.txt` completed successfully.

Extending the project
---------------------
- Add new discord.py commands or event handlers for richer interactions.
- Replace or extend the LLM integration to support streaming, alternate endpoints, or additional metadata.
- Introduce moderation or more advanced safety-filtering layers before sending replies.

Contribution and license
------------------------
Contributions are welcome. Follow standard Git workflows: branch, test, and open pull requests. See the LICENSE file for licensing details.
