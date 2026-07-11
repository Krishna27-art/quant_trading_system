#!/usr/bin/env python
import os
import sys
import time
import signal
import socket
import subprocess
from pathlib import Path

# Set up directories and environment variables
WORKSPACE_DIR = Path(__file__).parent.resolve()
LOGS_DIR = WORKSPACE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Set database URL and environment defaults
os.environ["DATABASE_URL"] = "sqlite:///quant.db"
os.environ["ENV"] = "LOCAL_DEV"

print("==================================================")
print("      INSTITUTIONAL QUANT SYSTEM LAUNCHER         ")
print("==================================================")

def kill_process_on_port(port):
    """Clean up any old processes running on the specified port."""
    try:
        # Get PIDs of processes running on the port
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                print(f"Killing process on port {port} (PID: {pid})...")
                subprocess.run(["kill", "-9", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        pass

# 1. Port Cleanup
print("[1/4] Cleaning up any old processes on ports 8000 and 5173...")
kill_process_on_port(8000)
kill_process_on_port(5173)

# 2. Daily Upstox Access Token Configuration
print("\n[2/4] Configuring Upstox Access Token...")
# Check command line arguments first
new_token = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    new_token = sys.argv[1]
    # Remove from sys.argv so it isn't passed to main.py
    sys.argv.pop(1)
else:
    # Check if we can prompt the user interactively
    if sys.stdin.isatty():
        try:
            print("Upstox access tokens expire daily. Please paste today's access token (or press Enter to keep current token):")
            new_token = input().strip()
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception:
            pass

if new_token:
    env_path = WORKSPACE_DIR / ".env"
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("UPSTOX_BROKER_ACCESS_TOKEN="):
                new_lines.append(f"UPSTOX_BROKER_ACCESS_TOKEN={new_token}")
            elif line.startswith("UPSTOX_ACCESS_TOKEN="):
                new_lines.append(f"UPSTOX_ACCESS_TOKEN={new_token}")
            else:
                new_lines.append(line)
        env_path.write_text("\n".join(new_lines) + "\n")
        print("Successfully updated .env with the new Upstox access token.")
    else:
        print("No .env file found. Skipping token write.")
else:
    print("Proceeding with existing .env token.")

# 3. Initialize Python interpreter path
python_executable = sys.executable

# 4. Start Servers
print("\n[3/4] Starting backend and frontend servers...")

# Start API Backend
backend_log = open(LOGS_DIR / "backend.log", "w")
print("Starting FastAPI Backend on port 8000...")
backend_proc = subprocess.Popen(
    [python_executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=backend_log,
    stderr=backend_log,
    cwd=str(WORKSPACE_DIR)
)

# Start Frontend UI Server
frontend_log = open(LOGS_DIR / "frontend.log", "w")
print("Starting Frontend UI on port 5173...")
frontend_proc = subprocess.Popen(
    ["npm", "run", "dev"],
    stdout=frontend_log,
    stderr=frontend_log,
    cwd=str(WORKSPACE_DIR / "frontend")
)

# Run initial prediction outcome resolver
print("\n[4/4] Running initial prediction outcome resolver...")
resolver_log = open(LOGS_DIR / "outcome_resolver.log", "w")
try:
    subprocess.run(
        [python_executable, "scripts/resolve_outcomes.py"],
        stdout=resolver_log,
        stderr=resolver_log,
        cwd=str(WORKSPACE_DIR),
        timeout=30
    )
except subprocess.TimeoutExpired:
    print("Initial outcome resolver timed out. Proceeding...")
except Exception as e:
    print(f"Failed to run initial outcome resolver: {e}")

# Start Outcome Resolver Loop in a background process
print("Starting background outcome resolver loop...")
resolver_proc = subprocess.Popen(
    [python_executable, "-c", "import time, subprocess, sys; \nwhile True:\n    time.sleep(3600)\n    subprocess.run([sys.executable, 'scripts/resolve_outcomes.py'])"],
    stdout=resolver_log,
    stderr=resolver_log,
    cwd=str(WORKSPACE_DIR)
)

# Keep track of processes to terminate them cleanly on exit
running_processes = [backend_proc, frontend_proc, resolver_proc]

def clean_shutdown(sig=None, frame=None):
    print("\nShutting down all servers...")
    for proc in running_processes:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    backend_log.close()
    frontend_log.close()
    resolver_log.close()
    sys.exit(0)

# Register shutdown signals
signal.signal(signal.SIGINT, clean_shutdown)
signal.signal(signal.SIGTERM, clean_shutdown)

print("\n" + "-" * 50)
print("System successfully launched!")
print("Backend API:  http://localhost:8000")
print("Frontend UI:  http://localhost:5173")
print("Backend Logs:  logs/backend.log")
print("Frontend Logs: logs/frontend.log")
print("Resolver Logs: logs/outcome_resolver.log")
print("Press Ctrl+C to stop all servers and exit.")
print("-" * 50 + "\n")

# Start Trading Orchestrator in the foreground
print("Starting Trading Orchestrator (Mode: paper)...")
orchestrator_args = [python_executable, "main.py", "--mode", "paper", "--duration", "28800"]
# Pass along any extra CLI arguments
orchestrator_args.extend(sys.argv[1:])

try:
    subprocess.run(orchestrator_args, cwd=str(WORKSPACE_DIR))
except KeyboardInterrupt:
    pass
finally:
    clean_shutdown()
