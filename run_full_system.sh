#!/bin/bash
set -e

echo "=================================================="
echo "      INSTITUTIONAL QUANT SYSTEM LAUNCHER         "
echo "=================================================="

# Ensure logs directory exists
mkdir -p logs

# 1. Port Cleanup (Foolproof port cleaning)
echo "[1/4] Cleaning up any old processes on ports 8000 and 3000..."
PID_8000=$(lsof -t -i:8000 || true)
if [ ! -z "$PID_8000" ]; then
    echo "Killing processes on port 8000: $PID_8000"
    kill -9 $PID_8000 2>/dev/null || true
fi

PID_3000=$(lsof -t -i:3000 || true)
if [ ! -z "$PID_3000" ]; then
    echo "Killing processes on port 3000: $PID_3000"
    kill -9 $PID_3000 2>/dev/null || true
fi

# 2. Daily Upstox Access Token Helper
echo ""
echo "[2/4] Configuring Upstox Access Token..."
# If token is passed as first argument, use it. Otherwise, prompt the user.
if [ ! -z "$1" ] && [[ "$1" != --* ]]; then
    NEW_TOKEN="$1"
    shift
    echo "Using token passed via command line argument."
else
    echo "Upstox access tokens expire daily. Please paste today's access token (or press Enter to keep current token):"
    read -r NEW_TOKEN
fi

if [ ! -z "$NEW_TOKEN" ]; then
    # Update .env file
    NEW_TOKEN="$NEW_TOKEN" python -c "
import os
token = os.environ['NEW_TOKEN']
with open('.env', 'r') as f:
    lines = f.readlines()
with open('.env', 'w') as f:
    for line in lines:
        if line.startswith('UPSTOX_BROKER_ACCESS_TOKEN='):
            f.write(f'UPSTOX_BROKER_ACCESS_TOKEN={token}\n')
        elif line.startswith('UPSTOX_ACCESS_TOKEN='):
            f.write(f'UPSTOX_ACCESS_TOKEN={token}\n')
        else:
            f.write(line)
"
    echo "Successfully updated .env with the new Upstox access token."
else
    echo "No token entered. Proceeding with existing .env token."
fi

# 3. Environment Activation
echo ""
echo "[3/4] Initializing Virtual Environment & DB..."
source .venv/bin/activate
export DATABASE_URL="sqlite:///quant.db"
export ENV="LOCAL_DEV"

# 4. Start Servers
echo ""
echo "[4/4] Starting servers..."

# Start API Backend
echo "Starting FastAPI Backend on port 8000..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
API_PID=$!

# Start Frontend UI Server
echo "Starting Frontend UI on port 3000..."
python -m http.server 3000 --directory frontend/dist > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Run prediction outcome resolver once immediately on startup
echo "Running initial prediction outcome resolver..."
python scripts/resolve_outcomes.py > logs/outcome_resolver.log 2>&1 || true

# Start Outcome Resolver loop in background (runs every hour to evaluate predictions)
(
  while true; do
    sleep 3600
    echo "Running prediction outcome resolver..."
    python scripts/resolve_outcomes.py >> logs/outcome_resolver.log 2>&1 || true
  done
) &
RESOLVER_PID=$!

# Cleanup trap to kill background processes on Exit
trap "echo ''; echo 'Shutting down servers...'; kill $API_PID $FRONTEND_PID $RESOLVER_PID 2>/dev/null || true; exit" INT TERM EXIT

echo ""
echo "--------------------------------------------------"
echo "System successfully launched!"
echo "Backend API:  http://localhost:8000"
echo "Frontend UI:  http://localhost:3000"
echo "Backend Logs:  logs/backend.log"
echo "Frontend Logs: logs/frontend.log"
echo "Resolver Logs: logs/outcome_resolver.log"
echo "Press Ctrl+C to stop all servers and exit."
echo "--------------------------------------------------"
echo ""

# Start Trading Orchestrator in foreground
echo "Starting Trading Orchestrator (Mode: paper)..."
python main.py --mode paper --duration 28800 "$@"
