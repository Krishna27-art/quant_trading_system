#!/bin/bash
echo "Starting Full Quant System..."

# Activate virtual environment
source .venv/bin/activate

# Set database URL for local development (SQLite with real data)
export DATABASE_URL="sqlite:///quant.db"

# Start API Backend in background
echo "Starting FastAPI Backend on port 8000..."
uvicorn api.main:app --port 8000 &
API_PID=$!

# Start Frontend Server in background
echo "Starting Frontend UI on port 3000..."
python -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

# Start Trading Orchestrator in foreground (so we can see logs)
echo "Starting Trading Orchestrator..."
python main.py --mode paper

# Cleanup trap for when orchestrator stops
trap "kill $API_PID $FRONTEND_PID" EXIT
