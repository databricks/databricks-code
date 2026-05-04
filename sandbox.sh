#!/usr/bin/env bash
# Run coding-gateway in a sandboxed HOME so it can't touch your real configs.
# Usage: ./sandbox.sh [any coding-gateway args...]
# Example: ./sandbox.sh configure
# Example: ./sandbox.sh --tool claude
# Example: ./sandbox.sh status

SANDBOX_HOME="/tmp/coding-gateway-sandbox"
mkdir -p "$SANDBOX_HOME"

echo "==> Sandboxed HOME: $SANDBOX_HOME"
echo "==> Your real ~/.claude, ~/.codex, ~/.gemini are untouched"
echo ""

HOME="$SANDBOX_HOME" uv run coding-gateway "$@"
