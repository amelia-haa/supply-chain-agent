# src/action/workflow_integrations.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class IntegrationResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IntegrationResult:
    status: IntegrationResultStatus
    provider: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class WorkflowIntegration(Protocol):
    """Integration adapter interface."""
    name: str

    async def notify(self, event_type: str, payload: Dict[str, Any]) -> IntegrationResult:
        ...


# -------------------------
# Utilities: retry/backoff
# -------------------------

async def _retry_async(
    fn,
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    retry_exceptions: Tuple[type, ...] = (aiohttp.ClientError, asyncio.TimeoutError),
):
    attempt = 0
    delay = base_delay
    while True:
        try:
            return await fn()
        except retry_exceptions as e:
            attempt += 1
            if attempt > retries:
                raise
            logger.warning(f"Retryable error: {e}. Retrying in {delay:.2f}s (attempt {attempt}/{retries})")
            await asyncio.sleep(delay)
            delay = min(max_delay, delay * 2)


def _make_idempotency_key(event_type: str, payload: Dict[str, Any]) -> str:
    """Create a stable key so integrations don’t duplicate notifications."""
    core = {
        "event_type": event_type,
        "company_id": payload.get("company_id"),
        "workflow_id": payload.get("workflow_id"),
        "task_id": payload.get("task_id"),
        "action_id": payload.get("action_id"),
    }
    return json.dumps(core, sort_keys=True)


# -------------------------
# Webhook integration
# -------------------------

class WebhookIntegration:
    """
    Generic webhook integration:
    - Works with Zapier/Make/n8n/custom endpoint
    - Adds an Idempotency-Key header
    """
    name = "webhook"

    def __init__(self, webhook_url: str, timeout_s: float = 8.0):
        self.webhook_url = webhook_url
        self.timeout_s = timeout_s

    async def notify(self, event_type: str, payload: Dict[str, Any]) -> IntegrationResult:
        if not self.webhook_url:
            return IntegrationResult(
                status=IntegrationResultStatus.SKIPPED,
                provider=self.name,
                message="No webhook_url configured",
            )

        idem_key = _make_idempotency_key(event_type, payload)
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idem_key,
            "X-Event-Type": event_type,
        }

        async def _call():
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json={"event_type": event_type, "payload": payload}, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise aiohttp.ClientError(f"Webhook error status={resp.status} body={text[:200]}")
                    return {"status": resp.status, "body": text[:500]}

        try:
            data = await _retry_async(_call, retries=3)
            return IntegrationResult(
                status=IntegrationResultStatus.SUCCESS,
                provider=self.name,
                message="Webhook delivered",
                data=data,
            )
        except Exception as e:
            return IntegrationResult(
                status=IntegrationResultStatus.FAILED,
                provider=self.name,
                message=f"Webhook delivery failed: {e}",
            )


# -------------------------
# Slack integration
# -------------------------

class SlackWebhookIntegration:
    """
    Slack Incoming Webhook integration.
    Pass slack_webhook_url in settings/env.
    """
    name = "slack"

    def __init__(self, slack_webhook_url: str, timeout_s: float = 8.0):
        self.slack_webhook_url = slack_webhook_url
        self.timeout_s = timeout_s

    async def notify(self, event_type: str, payload: Dict[str, Any]) -> IntegrationResult:
        if not self.slack_webhook_url:
            return IntegrationResult(
                status=IntegrationResultStatus.SKIPPED,
                provider=self.name,
                message="No slack_webhook_url configured",
            )

        # Minimal “good looking” Slack message.
        title = payload.get("title") or payload.get("workflow_name") or event_type
        level = payload.get("priority") or payload.get("status") or "info"
        workflow_id = payload.get("workflow_id")
        task_id = payload.get("task_id")

        text = (
            f"*{title}*\n"
            f"• Event: `{event_type}`\n"
            f"• Level: *{level}*\n"
            + (f"• Workflow: `{workflow_id}`\n" if workflow_id else "")
            + (f"• Task: `{task_id}`\n" if task_id else "")
        )

        async def _call():
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.slack_webhook_url, json={"text": text}) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise aiohttp.ClientError(f"Slack webhook failed status={resp.status} body={body[:200]}")
                    return {"status": resp.status, "body": body[:200]}

        try:
            data = await _retry_async(_call, retries=3)
            return IntegrationResult(
                status=IntegrationResultStatus.SUCCESS,
                provider=self.name,
                message="Slack message delivered",
                data=data,
            )
        except Exception as e:
            return IntegrationResult(
                status=IntegrationResultStatus.FAILED,
                provider=self.name,
                message=f"Slack delivery failed: {e}",
            )


# -------------------------
# Jira integration (stub)
# -------------------------

class JiraIntegration:
    """
    Jira integration stub.
    In real use: call Jira REST API to create issues.
    Here we just structure what a Jira payload WOULD look like.
    """
    name = "jira"

    def __init__(self, jira_base_url: str = "", project_key: str = ""):
        self.jira_base_url = jira_base_url
        self.project_key = project_key

    async def notify(self, event_type: str, payload: Dict[str, Any]) -> IntegrationResult:
        if not self.jira_base_url or not self.project_key:
            return IntegrationResult(
                status=IntegrationResultStatus.SKIPPED,
                provider=self.name,
                message="No Jira config (jira_base_url/project_key)",
            )

        jira_issue_payload = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": payload.get("title", f"{event_type}"),
                "description": json.dumps(payload, indent=2)[:5000],
                "issuetype": {"name": "Task"},
            }
        }

        # You can implement actual HTTP call later.
        return IntegrationResult(
            status=IntegrationResultStatus.SUCCESS,
            provider=self.name,
            message="Jira issue payload prepared (stub)",
            data={"jira_payload": jira_issue_payload},
        )