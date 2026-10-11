#!/usr/bin/env bash
# Starts MosImage on macOS (Apple silicon): the engine on http://127.0.0.1:8000
# and the UI on http://localhost:5173. The counterpart of start.cmd on Windows.
#   ./start.sh               set up what is missing, then start
#   ./start.sh --no-install  start with what is installed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$BACKEND/.venv"
PYTHON="$VENV/bin/python"
READY="$VENV/.mosimage-ready"
SUPPORTED="3.13 3.12"
NO_INSTALL=false
[[ "${1:-}" == "--no-install" ]] && NO_INSTALL=true

cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "start.sh targets macOS on Apple silicon; use start.cmd on Windows." >&2
    exit 1
fi

python_version() {
    "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

create_venv() {
    local version candidate
    for version in $SUPPORTED; do
        candidate="$(command -v "python$version" || true)"
        if [[ -n "$candidate" ]]; then
            echo "Creating backend/.venv with Python $version..."
            "$candidate" -m venv "$VENV"
            return
        fi
    done
    if command -v uv > /dev/null; then
        echo "Creating backend/.venv with Python 3.13 from uv..."
        uv venv --seed --python 3.13 "$VENV"
        return
    fi
    echo "MosImage needs Python 3.13 or 3.12 for PyTorch. Install one, for example with" >&2
    echo "  brew install python@3.13" >&2
    echo "then run ./start.sh again." >&2
    exit 1
}

metal_ready() {
    "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)' > /dev/null 2>&1
}

if [[ ! -x "$PYTHON" ]]; then
    if $NO_INSTALL; then
        echo "backend/.venv is missing; run ./start.sh without --no-install first." >&2
        exit 1
    fi
    create_venv
fi

version="$(python_version "$PYTHON")"
if [[ " $SUPPORTED " != *" $version "* ]]; then
    echo "backend/.venv uses Python $version, but PyTorch needs 3.13 or 3.12. Recreate it:" >&2
    echo "  rm -rf backend/.venv && ./start.sh" >&2
    exit 1
fi

if ! $NO_INSTALL; then
    needs_install=false
    [[ -f "$READY" ]] || needs_install=true
    if ! $needs_install && ! metal_ready; then
        echo "PyTorch in backend/.venv cannot use Metal. Reinstalling..."
        needs_install=true
    fi
    if $needs_install; then
        echo "Installing backend dependencies. The first run downloads about 1 GB..."
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -r "$BACKEND/requirements-torch-macos.txt"
        "$PYTHON" -m pip install -r "$BACKEND/requirements.txt"
        "$PYTHON" -c 'import torch; print("Torch:", torch.__version__); print("Metal (MPS) available:", torch.backends.mps.is_available())'
        metal_ready || { echo "PyTorch cannot use Metal on this Mac; MosImage needs macOS 14 or newer." >&2; exit 1; }
        touch "$READY"
    fi
    if [[ ! -d "$FRONTEND/node_modules" ]]; then
        echo "Installing frontend dependencies..."
        (cd "$FRONTEND" && npm install)
    fi
fi

if [[ ! -f "$BACKEND/.env" ]]; then
    cat > "$BACKEND/.env" <<'EOF'
# Local settings for this Mac. The defaults suit Apple silicon; .env.example lists every option.
# FLUX.2 Klein 9B (Quality) is gated: accept its licence on huggingface.co and add a read token,
# or log in once with: backend/.venv/bin/hf auth login
# MOSIMAGE_HF_TOKEN=hf_your_read_token
EOF
    echo "Created backend/.env"
fi

# Restarts the engine after a hard cancel or crash, like backend/supervisor.ps1.
supervise() {
    # The engine also exits by itself if this script ($$, also in this subshell) is killed outright.
    export MOSIMAGE_SUPERVISED=true MOSIMAGE_EXIT_WITH_PARENT=true MOSIMAGE_WATCH_PID=$$ PYTHONUNBUFFERED=1
    cd "$BACKEND"
    local code
    while true; do
        "$PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 && return
        code=$?
        # Asked to stop: uvicorn re-raises Ctrl+C (130) or SIGTERM (143) once it has shut down.
        [[ $code -eq 130 || $code -eq 143 ]] && return
        echo "MosImage engine exited with code $code. Restarting..." >&2
        sleep 1
    done
}

stop_all() {
    trap - EXIT INT TERM HUP
    echo "Stopping MosImage..."
    # Under set -e, a process that is already gone must not end this cleanup early.
    if [[ -n "${ui:-}" ]]; then kill -TERM "$ui" 2> /dev/null || true; fi
    local engine pid alive
    engine="$(pgrep -P "$supervisor" || true)"
    # The loop first, so it cannot start another engine; uvicorn unloads the models on SIGTERM.
    kill -TERM "$supervisor" 2> /dev/null || true
    for pid in $engine; do kill -TERM "$pid" 2> /dev/null || true; done
    for _ in $(seq 1 40); do
        alive=false
        for pid in $engine; do if kill -0 "$pid" 2> /dev/null; then alive=true; fi; done
        $alive || break
        sleep 0.5
    done
    exit 0
}

echo "Starting MosImage engine on http://127.0.0.1:8000"
supervise &
supervisor=$!
trap stop_all EXIT INT TERM HUP

sleep 2
echo "Starting MosImage UI on http://localhost:5173"
# In the background, so a signal reaches the trap at once instead of after the UI exits.
(cd "$FRONTEND" && exec ./node_modules/.bin/vite) &
ui=$!
wait "$ui"
