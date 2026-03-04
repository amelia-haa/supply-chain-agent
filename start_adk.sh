#!/usr/bin/env bash

# small helper script to start the ADK web server on the first available port
# starting from 8000.  It checks whether the port is in use and bumps if needed.
# usage: ./start_adk.sh [start-port]
# env:
#   APP_RUNTIME_MODE=api|vertex (default: api)
#   GOOGLE_API_KEY=... (required in api mode)
#   GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION / GOOGLE_APPLICATION_CREDENTIALS (required in vertex mode)

# Load local env files if present.
# Priority:
# 1) repo .env
# 2) agent/.env (legacy override)
MODE_OVERRIDE="${APP_RUNTIME_MODE:-}"
if [ -f "./.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "./.env"
    set +a
fi
if [ -f "./agent/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "./agent/.env"
    set +a
fi
if [ -n "${MODE_OVERRIDE}" ]; then
    APP_RUNTIME_MODE="${MODE_OVERRIDE}"
fi

MODE="${APP_RUNTIME_MODE:-api}"

if [ "$MODE" = "api" ]; then
    echo "runtime mode: api (Gemini API key)"
    # Force API mode and clear Vertex-only vars to avoid accidental backend switch.
    unset GOOGLE_CLOUD_PROJECT
    unset GOOGLE_CLOUD_LOCATION
    unset GOOGLE_APPLICATION_CREDENTIALS
    export GOOGLE_GENAI_USE_VERTEXAI=False
    if [ -z "${GOOGLE_API_KEY:-}" ]; then
        echo "ERROR: GOOGLE_API_KEY is required when APP_RUNTIME_MODE=api"
        exit 1
    fi
elif [ "$MODE" = "vertex" ]; then
    echo "runtime mode: vertex (Google Cloud Vertex AI)"
    export GOOGLE_GENAI_USE_VERTEXAI=True
    if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ] || [ -z "${GOOGLE_CLOUD_LOCATION:-}" ]; then
        echo "ERROR: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are required when APP_RUNTIME_MODE=vertex"
        exit 1
    fi
    # Accept either:
    # 1) explicit service-account json path via GOOGLE_APPLICATION_CREDENTIALS
    # 2) local Application Default Credentials (ADC) from gcloud auth login
    ADC_PATH="${HOME}/.config/gcloud/application_default_credentials.json"
    if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
        if [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
            echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS is set but file not found: ${GOOGLE_APPLICATION_CREDENTIALS}"
            exit 1
        fi
    elif [ ! -f "${ADC_PATH}" ]; then
        echo "ERROR: Vertex mode needs credentials. Set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'."
        exit 1
    fi
else
    echo "ERROR: APP_RUNTIME_MODE must be 'api' or 'vertex' (got: $MODE)"
    exit 1
fi

START_PORT=${1:-8000}
PORT=$START_PORT
AGENTS_DIR="./adk_apps"

while true; do
    # quick check for an existing listener
    if lsof -iTCP:$PORT -sTCP:LISTEN -P -n >/dev/null 2>&1; then
        echo "port $PORT is busy, trying next port"
        PORT=$((PORT+1))
        continue
    fi

    echo "attempting to start ADK web server on port $PORT..."
    # run ADK; if it fails with a bind error we loop and try the next port
    if adk web --port $PORT "$AGENTS_DIR"; then
        # server started and exited normally (user stopped it)
        exit 0
    else
        status=$?
        # Retry only when port is actually busy; otherwise exit and surface failure.
        if lsof -iTCP:$PORT -sTCP:LISTEN -P -n >/dev/null 2>&1; then
            echo "ADK failed on port $PORT and port is busy; trying next port"
            PORT=$((PORT+1))
        else
            echo "ADK exited with status $status on port $PORT; exiting"
            exit $status
        fi
    fi

    if [ $PORT -gt $((START_PORT+100)) ]; then
        echo "couldn't find a free port in range $START_PORT-$PORT"
        exit 1
    fi

done
