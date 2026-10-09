#!/usr/bin/env bash
# One-command launcher for macOS and Linux.
#   ./run.sh           demo mode, no training needed
#   ./run.sh trained   load the trained ensemble from artifacts/
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
fi
source .venv/bin/activate

if [ "$1" = "trained" ]; then
  pip install -q -r requirements.txt
  unset DEMO
else
  # Demo needs only the light dependencies, not torch or the boosting libraries.
  pip install -q flask flask-cors numpy
  export DEMO=1
fi

PORT="${PORT:-5050}"
echo ""
echo "  URL Inspector running at  http://localhost:$PORT"
echo "  Press Ctrl+C to stop."
echo ""
( sleep 2 && open "http://localhost:$PORT" 2>/dev/null || true ) &
PORT="$PORT" python -m backend.app
