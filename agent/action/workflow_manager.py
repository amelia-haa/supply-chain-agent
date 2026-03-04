# src/action/workflow_manager.py
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.config import settings
from agent.models import Company, Supplier, MitigationAction

from agent.action.workflow_integrations import (
    WorkflowIntegration,
    WebhookIntegration,
    SlackWebhookIntegration,
    JiraIntegration,
    IntegrationResult,
    IntegrationResultStatus,
)

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApprovalType(str, Enum):
    NONE = "none"
    HUMAN = "human"
    EXECUTIVE = "executive"


@dataclass
class WorkflowTask:
    task_id: str
    title: str
    description: str
    owner: str
    due_date: datetime
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    workflow_id: str
    workflow_name: str
    company_id: Any
    action_id: Any
    created_at: datetime
    status: WorkflowStatus = WorkflowStatus.PENDING
    approval_required: ApprovalType = ApprovalType.NONE
    tasks: List[WorkflowTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowManager:
    """
    Comprehensive Workflow Manager:
    - Create workflow templates
    - Track tasks + dependencies
    - Emit events to integrations (webhook/slack/jira)
    """

    def __init__(self):
        # In-memory store (replace with DB later)
        self._workflows: Dict[str, Workflow] = {}

        # Integrations
        self.integrations: List[WorkflowIntegration] = self._load_integrations()

    def _load_integrations(self) -> List[WorkflowIntegration]:
        integrations: List[WorkflowIntegration] = []

        # Webhook -> Zapier/Make/n8n/custom
        webhook_url = getattr(settings, "workflow_webhook_url", "") or ""
        integrations.append(WebhookIntegration(webhook_url))

        # Slack incoming webhook
        slack_url = getattr(settings, "slack_webhook_url", "") or ""
        integrations.append(SlackWebhookIntegration(slack_url))

        # Jira (stub)
        jira_base = getattr(settings, "jira_base_url", "") or ""
        jira_project = getattr(settings, "jira_project_key", "") or ""
        integrations.append(JiraIntegration(jira_base, jira_project))

        return integrations

    # -------------------------
    # Public: template creators
    # -------------------------

    async def create_supplier_qualification_workflow(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        wf = self._new_workflow(company, action, "Supplier Qualification Workflow", approval=ApprovalType.HUMAN)

        # Tasks (dependency graph)
        t1 = self._task("Identify candidate suppliers", "Shortlist 3–5 suppliers for the component/category.", "procurement_lead", days=7)
        t2 = self._task("Collect supplier documents", "Collect certifications, capacity, ESG, financials.", "procurement_analyst", days=14, depends=[t1.task_id])
        t3 = self._task("Quality + audit assessment", "Run quality audit and compliance checks.", "quality_manager", days=21, depends=[t2.task_id])
        t4 = self._task("Pilot order", "Place small pilot order and evaluate OTIF/quality.", "supply_chain_ops", days=35, depends=[t3.task_id])
        t5 = self._task("Approve onboarding", "Decision gate: approve supplier onboarding.", "supply_chain_director", days=40, depends=[t4.task_id], meta={"approval_gate": True})

        wf.tasks = [t1, t2, t3, t4, t5]
        self._workflows[wf.workflow_id] = wf

        await self._emit_event("workflow.created", self._workflow_payload(wf))
        await self._refresh_task_readiness(wf.workflow_id)

        return self._workflow_payload(wf)

    async def create_negotiation_workflow(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        wf = self._new_workflow(company, action, "Supplier Contract Negotiation Workflow", approval=ApprovalType.HUMAN)

        t1 = self._task("Prepare negotiation brief", "Collect spend, performance, risks, and target terms.", "procurement_lead", days=5)
        t2 = self._task("Draft term sheet", "Draft flexibility clauses, SLAs, penalties, dual-sourcing terms.", "legal_counsel", days=10, depends=[t1.task_id])
        t3 = self._task("Supplier negotiation meeting", "Conduct negotiation meeting and capture outcomes.", "procurement_lead", days=15, depends=[t2.task_id])
        t4 = self._task("Finalize contract redlines", "Finalize redlines, confirm compliance and approvals.", "legal_counsel", days=20, depends=[t3.task_id])
        t5 = self._task("Approve contract", "Decision gate: sign-off.", "finance_director", days=25, depends=[t4.task_id], meta={"approval_gate": True})

        wf.tasks = [t1, t2, t3, t4, t5]
        self._workflows[wf.workflow_id] = wf

        await self._emit_event("workflow.created", self._workflow_payload(wf))
        await self._refresh_task_readiness(wf.workflow_id)

        return self._workflow_payload(wf)

    async def create_escalation_workflow(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        # escalations often need executive approval
        approval = ApprovalType.EXECUTIVE if action.priority_level in ("critical", "high") else ApprovalType.HUMAN
        wf = self._new_workflow(company, action, "Supply Chain Escalation Workflow", approval=approval)

        t1 = self._task("Triage incident", "Confirm facts, scope, SKUs impacted, ETA, revenue-at-risk.", "risk_manager", days=1)
        t2 = self._task("Mitigation options", "Generate options: reroute, expedite, reallocate, buffer, substitute.", "supply_chain_planner", days=2, depends=[t1.task_id])
        t3 = self._task("Approval decision", "Decision gate: choose mitigation option.", "executive_owner", days=2, depends=[t2.task_id], meta={"approval_gate": True})
        t4 = self._task("Execute mitigation", "Create POs, reroute shipments, notify stakeholders.", "ops_manager", days=3, depends=[t3.task_id])
        t5 = self._task("Post-incident review", "Document root cause + preventive actions.", "risk_manager", days=7, depends=[t4.task_id])

        wf.tasks = [t1, t2, t3, t4, t5]
        self._workflows[wf.workflow_id] = wf

        await self._emit_event("workflow.created", self._workflow_payload(wf))
        await self._refresh_task_readiness(wf.workflow_id)

        return self._workflow_payload(wf)

    async def create_purchase_order_adjustment_workflow(self, company: Company, action: MitigationAction, po_ids: List[str]) -> Dict[str, Any]:
        wf = self._new_workflow(company, action, "Purchase Order Adjustment Workflow", approval=ApprovalType.HUMAN)
        wf.metadata["po_ids"] = po_ids

        t1 = self._task("Validate PO constraints", "Check MOQ, contract terms, supplier capacity, due dates.", "procurement_ops", days=2)
        t2 = self._task("Compute PO adjustment plan", "Create recommended changes: qty/date/split/expedite.", "planner", days=3, depends=[t1.task_id])
        t3 = self._task("Approval", "Decision gate: approve adjustments.", "procurement_manager", days=4, depends=[t2.task_id], meta={"approval_gate": True})
        t4 = self._task("Apply changes in ERP", "Submit approved changes via ERPIntegrator.", "erp_admin", days=5, depends=[t3.task_id])
        t5 = self._task("Notify suppliers + internal", "Send communications and update workflow notes.", "supply_chain_ops", days=6, depends=[t4.task_id])

        wf.tasks = [t1, t2, t3, t4, t5]
        self._workflows[wf.workflow_id] = wf

        await self._emit_event("workflow.created", self._workflow_payload(wf))
        await self._refresh_task_readiness(wf.workflow_id)

        return self._workflow_payload(wf)

    async def create_inventory_adjustment_workflow(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        wf = self._new_workflow(company, action, "Inventory Adjustment Workflow", approval=ApprovalType.HUMAN)

        t1 = self._task("Compute buffer targets", "Compute safety stock / reorder points by SKU.", "inventory_analyst", days=3)
        t2 = self._task("Carrying cost + space check", "Validate warehouse capacity and carrying costs.", "warehouse_manager", days=5, depends=[t1.task_id])
        t3 = self._task("Approval", "Decision gate: approve new buffer targets.", "ops_manager", days=6, depends=[t2.task_id], meta={"approval_gate": True})
        t4 = self._task("Update ERP parameters", "Write ROP/SS changes to ERP.", "erp_admin", days=7, depends=[t3.task_id])
        t5 = self._task("Monitor results", "Track service levels & inventory turns after change.", "inventory_manager", days=21, depends=[t4.task_id])

        wf.tasks = [t1, t2, t3, t4, t5]
        self._workflows[wf.workflow_id] = wf

        await self._emit_event("workflow.created", self._workflow_payload(wf))
        await self._refresh_task_readiness(wf.workflow_id)

        return self._workflow_payload(wf)

    async def create_general_workflow(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        wf = self._new_workflow(company, action, "General Mitigation Workflow", approval=ApprovalType.HUMAN)

        t1 = self._task("Define scope", "Define scope, success criteria, and owners.", "project_manager", days=3)
        t2 = self._task("Execute", "Execute mitigation action tasks.", "owner_team", days=14, depends=[t1.task_id])
        t3 = self._task("Review outcomes", "Confirm results and capture lessons learned.", "risk_manager", days=21, depends=[t2.task_id])

        wf.tasks = [t1, t2, t3]
        self._workflows[wf.workflow_id] = wf

        await self._emit_event("workflow.created", self._workflow_payload(wf))
        await self._refresh_task_readiness(wf.workflow_id)

        return self._workflow_payload(wf)

    # -------------------------
    # Public: task operations
    # -------------------------

    async def start_workflow(self, workflow_id: str) -> Dict[str, Any]:
        wf = self._require_workflow(workflow_id)
        wf.status = WorkflowStatus.IN_PROGRESS
        await self._emit_event("workflow.started", self._workflow_payload(wf))
        await self._refresh_task_readiness(workflow_id)
        return self._workflow_payload(wf)

    async def update_task_status(self, workflow_id: str, task_id: str, new_status: TaskStatus, note: str = "") -> Dict[str, Any]:
        wf = self._require_workflow(workflow_id)
        task = self._require_task(wf, task_id)

        task.status = new_status
        if note:
            task.metadata.setdefault("notes", []).append({"ts": datetime.utcnow().isoformat(), "note": note})

        await self._emit_event("task.updated", self._workflow_payload(wf, task_id=task_id))
        await self._refresh_task_readiness(workflow_id)
        await self._maybe_close_workflow(workflow_id)

        return self._workflow_payload(wf)

    async def approve_gate(self, workflow_id: str, task_id: str, approver: str, approved: bool, comment: str = "") -> Dict[str, Any]:
        wf = self._require_workflow(workflow_id)
        task = self._require_task(wf, task_id)

        if not task.metadata.get("approval_gate"):
            raise ValueError(f"Task {task_id} is not an approval gate")

        task.metadata["approval"] = {
            "approved": approved,
            "approver": approver,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        }
        task.status = TaskStatus.DONE if approved else TaskStatus.FAILED

        await self._emit_event("task.approval", self._workflow_payload(wf, task_id=task_id))
        await self._refresh_task_readiness(workflow_id)
        await self._maybe_close_workflow(workflow_id)

        return self._workflow_payload(wf)

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        wf = self._require_workflow(workflow_id)
        return self._workflow_payload(wf)

    def list_workflows(self) -> List[Dict[str, Any]]:
        return [self._workflow_payload(wf) for wf in self._workflows.values()]

    # -------------------------
    # Internals
    # -------------------------

    def _new_workflow(self, company: Company, action: MitigationAction, name: str, approval: ApprovalType) -> Workflow:
        wf_id = f"WF_{uuid.uuid4().hex[:10]}"
        return Workflow(
            workflow_id=wf_id,
            workflow_name=name,
            company_id=company.id,
            action_id=action.id,
            created_at=datetime.utcnow(),
            status=WorkflowStatus.PENDING,
            approval_required=approval,
            tasks=[],
            metadata={"priority": getattr(action, "priority_level", "medium")},
        )

    def _task(self, title: str, desc: str, owner: str, *, days: int, depends: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> WorkflowTask:
        return WorkflowTask(
            task_id=f"T_{uuid.uuid4().hex[:10]}",
            title=title,
            description=desc,
            owner=owner,
            due_date=datetime.utcnow() + timedelta(days=days),
            status=TaskStatus.PENDING,
            depends_on=depends or [],
            metadata=meta or {},
        )

    def _require_workflow(self, workflow_id: str) -> Workflow:
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow not found: {workflow_id}")
        return self._workflows[workflow_id]

    def _require_task(self, wf: Workflow, task_id: str) -> WorkflowTask:
        for t in wf.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task not found: {task_id}")

    async def _refresh_task_readiness(self, workflow_id: str) -> None:
        wf = self._require_workflow(workflow_id)

        # Mark tasks READY if all dependencies are DONE/SKIPPED
        done_ok = {TaskStatus.DONE, TaskStatus.SKIPPED}
        for t in wf.tasks:
            if t.status in (TaskStatus.DONE, TaskStatus.FAILED):
                continue

            deps = [self._require_task(wf, dep_id) for dep_id in t.depends_on]
            if all(d.status in done_ok for d in deps):
                if t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.READY
                    await self._emit_event("task.ready", self._workflow_payload(wf, task_id=t.task_id))
            else:
                if t.status in (TaskStatus.READY, TaskStatus.IN_PROGRESS):
                    t.status = TaskStatus.BLOCKED

        # If any task failed, workflow becomes BLOCKED/FAILED depending on policy
        if any(t.status == TaskStatus.FAILED for t in wf.tasks):
            wf.status = WorkflowStatus.BLOCKED

        if wf.status == WorkflowStatus.PENDING and any(t.status == TaskStatus.READY for t in wf.tasks):
            wf.status = WorkflowStatus.IN_PROGRESS

    async def _maybe_close_workflow(self, workflow_id: str) -> None:
        wf = self._require_workflow(workflow_id)

        if any(t.status == TaskStatus.FAILED for t in wf.tasks):
            wf.status = WorkflowStatus.FAILED
            await self._emit_event("workflow.failed", self._workflow_payload(wf))
            return

        if all(t.status in (TaskStatus.DONE, TaskStatus.SKIPPED) for t in wf.tasks):
            wf.status = WorkflowStatus.COMPLETED
            await self._emit_event("workflow.completed", self._workflow_payload(wf))

    async def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[IntegrationResult] = []
        for integ in self.integrations:
            try:
                res = await integ.notify(event_type, payload)
                results.append(res)
            except Exception as e:
                results.append(IntegrationResult(
                    status=IntegrationResultStatus.FAILED,
                    provider=getattr(integ, "name", "unknown"),
                    message=f"Integration crashed: {e}",
                ))

        # Store latest integration results on workflow metadata for debugging
        wf_id = payload.get("workflow_id")
        if wf_id and wf_id in self._workflows:
            self._workflows[wf_id].metadata["last_integration_results"] = [r.__dict__ for r in results]

        return [r.__dict__ for r in results]

    def _workflow_payload(self, wf: Workflow, task_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "workflow_id": wf.workflow_id,
            "workflow_name": wf.workflow_name,
            "company_id": wf.company_id,
            "action_id": wf.action_id,
            "status": wf.status.value,
            "approval_required": wf.approval_required.value,
            "created_at": wf.created_at.isoformat(),
            "priority": wf.metadata.get("priority", "medium"),
            "task_id": task_id,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "description": t.description,
                    "owner": t.owner,
                    "due_date": t.due_date.isoformat(),
                    "status": t.status.value,
                    "depends_on": t.depends_on,
                    "metadata": t.metadata,
                }
                for t in wf.tasks
            ],
            "metadata": wf.metadata,
        }