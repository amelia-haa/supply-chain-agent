#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./deploy_cloud_run.sh [PROJECT_ID] [REGION] [SERVICE_NAME]
#
# Example:
#   ./deploy_cloud_run.sh supply-chain-agent-489104 us-central1 supply-chain-agent-runner

PROJECT_ID="${1:-supply-chain-agent-489104}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-supply-chain-agent-runner}"

echo "Deploying Cloud Run service..."
echo "  project: ${PROJECT_ID}"
echo "  region:  ${REGION}"
echo "  service: ${SERVICE_NAME}"

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source . \
  --allow-unauthenticated \
  --set-env-vars "APP_RUNTIME_MODE=vertex,APP_SIGNAL_SOURCE=live,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GOOGLE_LLM_MODEL=gemini-2.0-flash,LIVE_INCLUDE_NOAA=true"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo "Deployed: ${SERVICE_URL}"
echo "Health:   ${SERVICE_URL}/healthz"
echo "Cycle:    ${SERVICE_URL}/run-cycle"
