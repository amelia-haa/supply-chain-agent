# Autonomous Supply Chain Resilience Agent

AI co-pilot prototype for mid-market manufacturing stability, built with Google ADK + Gemini.

## 1) Prerequisites

- Python 3.10+ (tested on Python 3.13)
- `pip`
- Optional for Vertex mode: `gcloud` CLI
- Internet access only if you want live feed ingestion

## 2) Clone and install

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Configure environment

Create local env from template:

```bash
cp .env.example .env
```

Edit `.env` and choose one runtime mode:

### Option A: API mode (recommended for demo speed)

```bash
APP_RUNTIME_MODE=api
GOOGLE_API_KEY=<your_google_ai_studio_api_key>
```

### Option B: Vertex mode (Google Cloud)

```bash
APP_RUNTIME_MODE=vertex
GOOGLE_CLOUD_PROJECT=<your_project_id>
GOOGLE_CLOUD_LOCATION=us-central1
```

Auth for Vertex mode (pick one):

- `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json`
- or ADC:
  - `gcloud auth application-default login`

Load env in terminal:

```bash
set -a; source .env; set +a
```

## 4) Run the agent (CLI)

Single cycle:

```bash
python3 main.py --cycles 1
```

Multiple companies:

```bash
python3 main.py --cycles 1 --companies de_semiconductor_auto,mx_multisource_industrial
```

Continuous loop:

```bash
python3 main.py --cycles 24 --interval-seconds 60
```

## 5) Run in Google ADK Web UI

```bash
./start_adk.sh
```

Open:

- `http://127.0.0.1:8000`

The app is packaged as one ADK app:

- `adk_apps/autonomous_supply_chain_agent`

## 6) Data modes

Set `APP_SIGNAL_SOURCE`:

- `mock`: `data/disruption_signals.json` (default)
- `live`: `data/live_disruption_signals.json` (falls back to mock if live is empty)
- `hybrid`: merge mock + live

Refresh live feed manually:

```bash
./refresh_live_data.sh 40
```

or:

```bash
python3 live_ingest.py --max-items 40
```

## 6.1) Real workflow execution integrations (optional)

To move from draft-only actions to real outbound workflow execution, set:

```bash
WORKFLOW_WEBHOOK_URL=<your webhook endpoint>
SLACK_WEBHOOK_URL=<your incoming slack webhook>
```

When set, each cycle attempts real webhook delivery with retries and logs delivery status.

## 7) Critical escalation demo mode

```bash
export APP_SIGNAL_PROFILE=critical
python3 main.py --cycles 1 --companies de_semiconductor_auto
```

## 8) Tests and dashboard

Run tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Generate dashboard artifacts:

```bash
python3 generate_dashboard.py
```

Outputs:

## 9) What this prototype demonstrates

- Real-time-ready disruption monitoring flow (mock/live/hybrid input)
- Multi-step reasoning pipeline with cost controls
- Mitigation planning and action generation
- Memory write-back and reflection loop
- Optional real webhook workflow execution (with retries + status logs)
- Hyper-personalized profile-based analysis
- Dual runtime mode (`api` and `vertex`)
- Judge-ready scorecard outputs (`business_impact_report`, `judging_scorecard`)

Board demo command (in ADK chat):

```text
Run the board demo and return headline score plus the best mitigation case.
```

Additional advanced demo prompts:

```text
Run a what-if simulation for de_semiconductor_auto with fuel_multiplier=1.2, lead_time_shock_days=5, demand_shock_pct=10.
```

```text
Onboard a new company profile: name=Acme Motion, region=Canada, industry=industrial_components, components=controllers,bearings, risk_appetite=low.
```

```text
Generate ROI benchmark report for de_semiconductor_auto,mx_multisource_industrial.
```

## 10) Public GitHub safety checklist

Before pushing:

1. Ensure no secrets are committed (`.env`, `agent/.env`, service-account JSON).
2. Rotate any key that was ever pasted in chat, screenshots, or terminal history.
3. Keep only `.env.example` in repo.
4. Verify `.gitignore` includes `.env`, `*.env`, `.adk/`, and key files.

Secret scan:

```bash
rg -n "AIza|GOOGLE_API_KEY|BEGIN PRIVATE KEY|-----BEGIN" .
```

## 11) Copy-safe handoff checklist

If you copy this project into another repo, include:

- `adk_apps/`
- `agent/`
- `data/`
- `deliverables/`
- `tests/`
- `main.py`, `start_adk.sh`, `refresh_live_data.sh`, `live_ingest.py`, `generate_dashboard.py`
- `requirements.txt`, `.env.example`, `.gitignore`, `README.md`

Do not copy:

- `.env`
- `agent/.env`
- Any service-account JSON key file
- `__pycache__/`, `.DS_Store`, `.adk/`

## 12) Troubleshooting

- `GOOGLE_API_KEY is required`: set API key and `APP_RUNTIME_MODE=api`.
- Vertex auth error: run `gcloud auth application-default login` or set valid `GOOGLE_APPLICATION_CREDENTIALS`.
- ADK port busy: `start_adk.sh` auto-increments from 8000.
- Empty live feed: run `./refresh_live_data.sh 40`, then rerun agent.

## 13) Event-Driven Cloud Run Deployment

This project includes a Cloud Run HTTP service that only runs full cycles when new disruption signals appear.

Files:

- `cloud_run_app.py`
- `Dockerfile`
- `.dockerignore`
- `deploy_cloud_run.sh`

Deploy:

```bash
gcloud auth login
gcloud config set project supply-chain-agent-489104
./deploy_cloud_run.sh supply-chain-agent-489104 us-central1 supply-chain-agent-runner
```

Test:

```bash
curl <SERVICE_URL>/healthz
curl -X POST <SERVICE_URL>/run-cycle -H "Content-Type: application/json" -d '{"company_ids":["de_semiconductor_auto","mx_multisource_industrial"],"max_items":40}'
```

Expected behavior:

- If no new signals: returns `status=skipped_no_new_signals` (no expensive run)
- If new signals: returns `status=processed` with cycle output

Create Cloud Scheduler (every 15 min):

```bash
gcloud scheduler jobs create http supply-chain-agent-schedule \
  --location=us-central1 \
  --schedule="*/15 * * * *" \
  --uri="<SERVICE_URL>/run-cycle" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"company_ids":["de_semiconductor_auto","mx_multisource_industrial"],"max_items":40}' \
  --oidc-service-account-email="<SCHEDULER_SA_EMAIL>"
```
