#!/bin/bash
# Send a Telegram message.
#   Usage: ./notify_telegram.sh "your message"
# Reads credentials from ~/.telegram_creds (NOT committed to git):
#   TELEGRAM_TOKEN=123456:ABC...
#   TELEGRAM_CHAT=987654321
# Fails quietly (never breaks a build/watcher) if creds are missing.

CREDS="$HOME/.telegram_creds"
[ -f "$CREDS" ] && source "$CREDS"

if [ -z "$TELEGRAM_TOKEN" ] || [ -z "$TELEGRAM_CHAT" ]; then
    echo "notify_telegram: no creds in ~/.telegram_creds; skipping" >&2
    exit 0
fi

curl -s -m 15 "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT}" \
    -d text="$1" >/dev/null 2>&1
