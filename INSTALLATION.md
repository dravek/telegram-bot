# Installation Guide

Step-by-step instructions for a junior developer to get the bot running from scratch.

---

## 1. Create a Telegram Bot Token

1. Open Telegram and search for **@BotFather**.
2. Send `/newbot` and follow the prompts (choose a name and username).
3. BotFather will reply with a token like `123456789:AAF...xyz`. **Copy it.**

---

## 2. Get an AI Provider API Key

**Option A — OpenAI:**
1. Go to <https://platform.openai.com/api-keys>.
2. Click **Create new secret key**, give it a name, copy the value.
3. Make sure your account has credits / an active plan.

**Option B — Anthropic:**
1. Go to <https://console.anthropic.com/settings/keys>.
2. Click **Create Key**, copy the value.

---

## 3. Set Up Python

Requires Python 3.12 or later.

```bash
# Check your version
python3 --version

# If needed, install Python 3.12 (Ubuntu/Debian example)
sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-pip
```

---

## 4. Clone / Download the Code

```bash
git clone <repository-url>
cd telegram-bot
```

---

## 5. Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

To also install development/test dependencies:

```bash
pip install -e ".[dev]"
```

---

## 6. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your values:

```dotenv
TELEGRAM_BOT_TOKEN=123456789:AAF...xyz

LLM_PROVIDER=openai          # or: anthropic

OPENAI_API_KEY=sk-...        # leave blank if using Anthropic
OPENAI_MODEL=gpt-4o-mini     # optional, this is the default

ANTHROPIC_API_KEY=sk-ant-... # leave blank if using OpenAI
ANTHROPIC_MODEL=claude-3-5-haiku-latest  # optional

MEMORY_SIZE=10               # optional, how many messages to remember
```

> **Never commit `.env` to version control.** It is already in `.gitignore`.

---

## 7. Start the Bot

```bash
# Load the .env variables into the current shell session
set -a; source .env; set +a

# Run
python app.py
```

You should see output like:

```
2024-01-15T12:00:00 INFO     __main__: Bot starting — provider=openai model=gpt-4o-mini memory_size=10
```

Open Telegram, find your bot, and send `/start` to verify it responds.

---

## 8. Run the Tests (optional)

```bash
pytest
```

All tests should pass without any live API keys since they use mocks.

---

## Next Steps

- To keep the bot running after you close the terminal, see **[SERVER.md](SERVER.md)**.
- To switch AI providers, change `LLM_PROVIDER` (and set the corresponding key) then restart.
