# Autonomous Supply Chain Resilience Agent -- LIBRA 

AI Co-Pilot for Mid-Market Manufacturing Stability.  
This project delivers a working autonomous agent that monitors disruptions, scores risk, simulates mitigation trade-offs, generates actions, and learns from past cycles.

---

## 1) What This Agent Does

The agent runs an end-to-end resilience cycle:

1. **Perception**
   - Ingest disruption signals (mock / live / hybrid)
   - Filter and classify disruption events

2. **Risk Intelligence**
   - Assess operational exposure across:
     - procurement
     - logistics
     - inventory
   - Compute risk score + reasons
   - Estimate revenue-at-risk

3. **Planning**
   - Simulate trade-offs between:
     - cost
     - service protection
     - resilience uplift
   - Rank mitigation options

4. **Action**
   - Draft supplier negotiation email
   - Flag ERP reorder adjustments
   - Recommend pre-emptive stock builds
   - Trigger escalation/workflow actions

5. **Memory & Reflection**
   - Write cycle outputs to memory
   - Reuse historical outcomes to influence future recommendations

---

## 2) Key Capabilities Mapped to Hackathon Requirements

- Real-time-ready disruption monitoring flow
- Multi-step reasoning pipeline
- Mitigation planning with trade-off simulation
- Action generation (supplier / ERP / escalation)
- Memory and feedback learning
- Hyper-personalized logic by company profile
- Proactive trigger-based autonomy with configurable human oversight
- Transparent decision traces and responsible-AI guardrails
- Cost-optimized deterministic-first architecture (LLM only when needed)

---

## 3) Repository Structure

```text
my_agent/
├─ adk_apps/
│  └─ autonomous_supply_chain_agent/
├─ agent/
│  ├─ agent.py
│  ├─ orchestrator.py
│  ├─ tools.py
│  ├─ autonomous_loop.py
│  ├─ memory.json
│  ├─ pipeline_cache.json
│  ├─ event_state.json
│  ├─ workflow_execution_log.json
│  └─ drift_state.json
├─ data/
│  ├─ company_profiles.json
│  ├─ disruption_signals.json
│  ├─ disruption_signals_critical.json
│  └─ live_disruption_signals.json
├─ tests/
│  └─ test_agent_pipeline.py
├─ cloud_run_app.py
├─ main.py
├─ live_ingest.py
├─ refresh_live_data.sh
├─ start_adk.sh
├─ deploy_cloud_run.sh
├─ Dockerfile
├─ requirements.txt
└─ .env.example
```

---

## 4) Prerequisites

- Python 3.10+ (tested on 3.13 locally)
- `pip`
- Optional: `gcloud` CLI for Vertex/Cloud Run
- Internet access only if using live signal ingestion

---

## 5) Installation

```bash
git clone <your-repo-url>
cd my_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6) Environment Configuration

Create `.env` from template:

```bash
cp .env.example .env
```

### Required minimum (API mode)
```env
APP_RUNTIME_MODE=api
GOOGLE_API_KEY=<your_google_api_key>
GOOGLE_LLM_MODEL=gemini-2.5-flash
```

### Optional (Vertex mode)
```env
APP_RUNTIME_MODE=vertex
GOOGLE_CLOUD_PROJECT=<your_project_id>
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

### Core runtime switches
```env
APP_DEMO_MODE=false
APP_SIGNAL_SOURCE=mock         # mock | live | hybrid
APP_SIGNAL_PROFILE=default     # default | critical
APP_AUTONOMY_MODE=human_approve # assistive | human_approve | auto_execute
APP_TRIGGER_REVENUE_AT_RISK_USD=500000
APP_TRIGGER_SUPPLIER_HEALTH_DROP=0.15
APP_ESCALATION_SLA_HOURS=6
LIVE_INCLUDE_NOAA=true
WORKFLOW_WEBHOOK_URL=
SLACK_WEBHOOK_URL=
```

Load env:
```bash
set -a; source .env; set +a
```

---

## 7) Run Options

### 7.1 CLI (single/multi cycle)

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

### 7.2 ADK Web UI

```bash
./start_adk.sh
```

Open:
- `http://127.0.0.1:8000`

---

## 8) Data Modes

- `APP_SIGNAL_SOURCE=mock`: use static mock signals
- `APP_SIGNAL_SOURCE=live`: use live signals file (fallback to mock if empty)
- `APP_SIGNAL_SOURCE=hybrid`: combine live + mock

Refresh live feed manually:
```bash
python3 live_ingest.py --max-items 40
# or
./refresh_live_data.sh 40
```

---

## 9) Demo Prompt Pack

### 9.1 Full feature cycle (single company)
```text
Run one full autonomous cycle for de_semiconductor_auto. Show monitoring summary, risk exposure, top 3 trade-offs, generated actions, proactive triggers fired, and memory update.
```

### 9.2 Hyper-personalization comparison
```text
Run a side-by-side full cycle for de_semiconductor_auto and mx_multisource_industrial with include_full_output=true. Compare supplier concentration, regional exposure, lead-time sensitivity, inventory policy, contract structures, SLA constraints, risk reasons, and top actions.
```

### 9.3 Proactive autonomy proof
```text
Run one autonomous cycle for de_semiconductor_auto and return a Proactiveness Report:
early-warning detection, trigger rules fired, workflows initiated automatically, supplier email draft, ERP reorder flags, pre-emptive stock recommendation, and escalation decision with threshold evidence.
```

### 9.4 Business impact framing
```text
Reframe business impact in two horizons: immediate cycle cash impact vs 12-month resilience value.
```

---

## 10) Testing

Run unit tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Expected:
- 10 tests passing

---

## 11) Cloud Run Deployment (Event-Driven)

Deploy:
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
./deploy_cloud_run.sh <PROJECT_ID> us-central1 supply-chain-agent-runner
```

Health check:
```bash
curl <SERVICE_URL>/healthz
```

Run cycle:
```bash
curl -X POST <SERVICE_URL>/run-cycle \
  -H "Content-Type: application/json" \
  -d '{"company_ids":["de_semiconductor_auto","mx_multisource_industrial"],"max_items":40}'
```

Expected statuses:
- `processed`
- `skipped_no_new_signals`
- `deferred_quota` (quota-aware fallback)

---

## 12) Cloud Scheduler (every 15 min example)

```bash
gcloud scheduler jobs create http supply-chain-agent-schedule \
  --location=us-central1 \
  --schedule="*/15 * * * *" \
  --uri="<SERVICE_URL>/run-cycle" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"company_ids":["de_semiconductor_auto","mx_multisource_industrial"],"max_items":40}'
```

---

## 13) Proactive Trigger Logic (Implemented)

The runtime evaluates explicit trigger rules:
- severity >= 0.8 (4/5)
- revenue_at_risk >= `APP_TRIGGER_REVENUE_AT_RISK_USD`
- inventory buffer coverage < projected delay
- supplier health drop >= `APP_TRIGGER_SUPPLIER_HEALTH_DROP`

When fired, agent can automatically:
- draft supplier negotiation email
- strengthen ERP reorder recommendation
- recommend stock build
- escalate to executives based on policy/thresholds

---

## 14) Execution Modes (Human Oversight)

- `assistive`: draft-only
- `human_approve` (default): elevated actions require approval
- `auto_execute`: allows autonomous execution if threshold + appetite conditions pass

Configured via:
```env
APP_AUTONOMY_MODE=assistive|human_approve|auto_execute
```

---

## 15) Responsible AI & Controls

- Human-in-the-loop override thresholds
- Budget guardrails
- Supplier concentration checks
- Regional alignment checks
- Transparency trace with decision rationale
- Quota/degradation-safe behavior

---

## 16) Security & Git Hygiene

Do **not** commit:
- `.env`
- service account JSON keys
- any private credentials

Quick secret scan:
```bash
rg -n "AIza|GOOGLE_API_KEY|BEGIN PRIVATE KEY|-----BEGIN|service-account" .
```

---

## 17) Troubleshooting

- **`GOOGLE_API_KEY is required`**
  - set `APP_RUNTIME_MODE=api` and `GOOGLE_API_KEY`
- **Vertex auth errors**
  - set `GOOGLE_APPLICATION_CREDENTIALS` or run `gcloud auth application-default login`
- **ADK port busy**
  - `start_adk.sh` auto-increments from 8000
- **Quota 429**
  - use Vertex/billing-backed mode, reduce run frequency, rely on event-driven gating
- **No live signals**
  - run `python3 live_ingest.py --max-items 40` and use `APP_SIGNAL_SOURCE=live|hybrid`
- **Agent gives concise output only**
  - set `APP_DEMO_MODE=false`

---

## 18) Tech Stack Summary

- Agent orchestration: Google ADK
- Model: Gemini (`GOOGLE_LLM_MODEL`)
- Backend: Python, FastAPI, Pydantic, Uvicorn
- Cloud: Google Cloud Run + Cloud Scheduler
- Data/state: JSON structured files
- Integrations: webhook + Slack webhook (optional)
- Testing: Python `unittest`
- Container: Docker

---

## 19) License / Submission Notes

If this is for hackathon submission:
- include YouTube demo link
- include PPTX link
- include this GitHub repo link
- include a clear requirement-to-feature mapping slide