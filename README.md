# Telegram AI Bot

A minimal, production-ready Telegram bot that chats using either **OpenAI** or **Anthropic** as the AI backend — configured entirely through environment variables.

## Features

- 💬 Natural language chat with per-chat conversation memory
- 🤖 Supports **OpenAI** (`gpt-4o-mini` by default) and **Anthropic** (`claude-3-5-haiku-latest` by default)
- 🔁 Automatic retry with exponential back-off for transient API errors
- 🧠 Sliding-window memory per chat (configurable, default: 10 messages)
- 📟 Commands: `/start`, `/help`, `/ping`, `/provider`, `/reset`
- 🪵 Structured logging to stdout
- 🐍 Python 3.12+, minimal dependencies

## Quickstart

```bash
git clone <repo>
cd telegram-bot

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN, LLM_PROVIDER, and the matching API key

# Load env vars and start
set -a; source .env; set +a
python app.py
```

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Greeting and configuration hint |
| `/help` | List all commands |
| `/ping` | Liveness check + uptime |
| `/provider` | Show active provider and model |
| `/reset` | Clear conversation memory for this chat |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | — | Bot token from @BotFather |
| `LLM_PROVIDER` | ✅ | — | `openai` or `anthropic` |
| `OPENAI_API_KEY` | ✅ if openai | — | OpenAI secret key |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI chat model |
| `ANTHROPIC_API_KEY` | ✅ if anthropic | — | Anthropic secret key |
| `ANTHROPIC_MODEL` | ❌ | `claude-3-5-haiku-latest` | Anthropic model |
| `MEMORY_SIZE` | ❌ | `10` | Messages kept per chat |

## Project Structure

```
telegram-bot/
├── app.py                          # Entry point
├── bot.py                          # Telegram handlers & Application factory
├── config.py                       # Env-var loading + validation
├── memory.py                       # Per-chat sliding-window memory
├── providers/
│   ├── __init__.py
│   ├── base.py                     # Abstract provider interface
│   ├── openai_provider.py          # OpenAI wrapper
│   └── anthropic_provider.py      # Anthropic wrapper
├── tests/
│   ├── test_config.py
│   ├── test_memory.py
│   └── test_providers.py
├── pyproject.toml
├── .env.example
├── README.md
├── INSTALLATION.md
└── SERVER.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Running on a Server

See [INSTALLATION.md](INSTALLATION.md) for a step-by-step setup guide and [SERVER.md](SERVER.md) for systemd configuration and production hardening.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `TELEGRAM_BOT_TOKEN is not set` | Missing env var | Set the variable in `.env` |
| `LLM_PROVIDER must be 'openai' or 'anthropic'` | Wrong/missing value | Set `LLM_PROVIDER=openai` or `=anthropic` |
| `OPENAI_API_KEY must be set` | Missing key for chosen provider | Add the key to `.env` |
| `Unauthorized` from Telegram | Invalid bot token | Re-generate the token with @BotFather |
| `RateLimitError` from provider | Exceeded API quota | The bot retries automatically; check your plan |
| Bot receives no messages | Pending updates dropped on restart | Normal — use `/start` to begin a new session |
| `PermissionError` (403) from provider | Model access not granted to key | Check your API plan / enable the model |
