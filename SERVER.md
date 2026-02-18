# Running on a Server

This guide covers deploying the bot as a long-running systemd service on a Linux server (Ubuntu / Debian).

---

## 1. Create a Dedicated User

Running the bot as a non-root user limits the blast radius of any security issue.

```bash
sudo useradd --system --create-home --shell /bin/bash botuser
sudo su - botuser
```

---

## 2. Deploy the Code

```bash
# As botuser
git clone <repository-url> ~/telegram-bot
cd ~/telegram-bot

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## 3. Create the Environment File

```bash
cp .env.example /home/botuser/telegram-bot/.env
chmod 600 /home/botuser/telegram-bot/.env   # only owner can read
```

Edit `/home/botuser/telegram-bot/.env` with your real credentials.

---

## 4. Create the systemd Service File

Create `/etc/systemd/system/telegram-bot.service` (as root):

```ini
[Unit]
Description=Telegram AI Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/home/botuser/telegram-bot
EnvironmentFile=/home/botuser/telegram-bot/.env
ExecStart=/home/botuser/telegram-bot/.venv/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
# Prevent runaway restarts
StartLimitInterval=60s
StartLimitBurst=5

# Basic hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/home/botuser/telegram-bot

[Install]
WantedBy=multi-user.target
```

---

## 5. Enable and Start the Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot
sudo systemctl status telegram-bot
```

---

## 6. Viewing Logs

```bash
# Live log stream
sudo journalctl -u telegram-bot -f

# Last 100 lines
sudo journalctl -u telegram-bot -n 100

# Logs since a specific time
sudo journalctl -u telegram-bot --since "2024-01-15 12:00:00"
```

---

## 7. Updating the Bot

```bash
sudo su - botuser
cd ~/telegram-bot
git pull

# If dependencies changed:
source .venv/bin/activate
pip install -e .

exit  # back to sudo user

sudo systemctl restart telegram-bot
sudo systemctl status telegram-bot
```

---

## 8. Restart Policies

The service file uses `Restart=always` with `RestartSec=10`, meaning systemd waits 10 seconds before restarting after a crash. `StartLimitBurst=5` prevents a tight crash loop — if the bot fails 5 times within 60 seconds systemd stops retrying (use `systemctl reset-failed telegram-bot` to clear this).

---

## 9. Basic Hardening Tips

| Tip | Detail |
|---|---|
| Run as non-root | Done — dedicated `botuser` account |
| Restrictive file permissions | `.env` is `chmod 600`; only owner can read |
| `NoNewPrivileges=true` | Process cannot gain extra privileges |
| `PrivateTmp=true` | Isolated `/tmp` directory |
| `ProtectSystem=strict` | Filesystem is read-only except `ReadWritePaths` |
| Keep secrets out of git | `.env` is in `.gitignore` |
| Rotate API keys periodically | Revoke and reissue keys from the provider consoles |

---

## 10. Health Check (Optional)

The bot does not expose an HTTP port, so a simple process-level check is sufficient:

```bash
# Check the service is active
systemctl is-active telegram-bot

# From a cron job (runs every 5 minutes, restarts if not active)
*/5 * * * * systemctl is-active --quiet telegram-bot || systemctl restart telegram-bot
```

For a more robust watchdog, systemd's built-in `WatchdogSec` can be enabled by adding notifications to `app.py` via the `sdnotify` package — this is optional for most deployments.

---

## 11. Log Rotation

Logs are written to the systemd journal, which handles rotation automatically. To check the journal disk usage:

```bash
journalctl --disk-usage
# Optionally limit journal size in /etc/systemd/journald.conf:
# SystemMaxUse=200M
```
