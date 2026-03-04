"""
Decision Tree Reasoning (Planning & Decision Engine)

Purpose
- Turn risk + scenario signals into a *traceable*, *auditable* decision path:
  "IF risk is high AND service impact is severe THEN trigger supplier reallocation + buffer stock + exec approval"

What you get
- A small decision-tree engine that:
  1) Builds a decision tree from your inputs (risk_assessment, scenarios, etc.)
  2) Evaluates it (async) to produce:
     - recommended actions
     - approval level (auto / human review / exec)
     - rationale trace (every rule that fired)
     - confidence score

How to use
- Call `await DecisionTreeReasoner().reason(...)`
- Feed outputs into your PlanningEngine/agent to decide what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class DecisionContext:
    """All signals the tree can use."""
    company_id: str
    timestamp: datetime
    risk_assessment: Dict[str, Any]
    scenarios: List[Dict[str, Any]]
    # optional extras your agent may provide
    buffer_policy: Optional[Dict[str, Any]] = None
    supplier_reallocation: Optional[Dict[str, Any]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutcome:
    """What the tree returns."""
    approval_level: str                      # "auto", "human_review", "executive"
    recommended_actions: List[Dict[str, Any]] # machine-friendly actions
    decision_trace: List[Dict[str, Any]]      # every fired rule/path
    decision_score: float                     # 0..1 severity/priority score
    confidence: float                         # 0..1
    metadata: Dict[str, Any] = field(default_factory=dict)


# A decision node can be:
# - Condition node: if predicate(context) then go left else right
# - Leaf node: emits actions + approval recommendation
@dataclass
class DecisionNode:
    node_id: str
    description: str

    predicate: Optional[Callable[[DecisionContext], bool]] = None
    true_child: Optional["DecisionNode"] = None
    false_child: Optional["DecisionNode"] = None

    # Leaf payload
    leaf_actions: Optional[List[Dict[str, Any]]] = None
    leaf_approval: Optional[str] = None
    leaf_score: Optional[float] = None

    def is_leaf(self) -> bool:
        return self.predicate is None


# ----------------------------
# Utility helpers
# ----------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _pick_scenario(scenarios: List[Dict[str, Any]], disruption_type: str) -> Optional[Dict[str, Any]]:
    """Find the first disruption scenario matching a type."""
    for sc in scenarios:
        if sc.get("scenario_type") == "disruption" and sc.get("disruption_type") == disruption_type:
            return sc
    return None


def _scenario_service_impact(sc: Optional[Dict[str, Any]]) -> float:
    """
    Normalize service impact to 0..1 if present.
    We accept different keys because your code varies across modules.
    """
    if not sc:
        return 0.0
    impact = sc.get("impact_analysis", {})
    # common keys you might have:
    # - impact_analysis["service_level_impact"] (0..1)
    # - impact_analysis["operational_impact"]["service_level_drop"] (0..1)
    if isinstance(impact, dict):
        if "service_level_impact" in impact:
            return _clamp(_safe_float(impact.get("service_level_impact"), 0.0))
        op = impact.get("operational_impact")
        if isinstance(op, dict) and "service_level_drop" in op:
            return _clamp(_safe_float(op.get("service_level_drop"), 0.0))
    # fallback
    if "service_level_impact" in sc:
        return _clamp(_safe_float(sc.get("service_level_impact"), 0.0))
    return 0.0


def _scenario_financial_impact(sc: Optional[Dict[str, Any]]) -> float:
    """
    Normalize financial impact to a proxy 0..1 using thresholds.
    If you already compute $ impacts, we map:
      <= 250k -> 0.2
      <= 1M   -> 0.5
      <= 5M   -> 0.8
      >  5M   -> 1.0
    """
    if not sc:
        return 0.0
    impact = sc.get("impact_analysis", {})
    value = None

    # common possibilities
    if isinstance(impact, dict):
        if "financial_impact" in impact:
            value = impact.get("financial_impact")
        elif "revenue_at_risk" in impact:
            value = impact.get("revenue_at_risk")

    if value is None and "financial_impact" in sc:
        value = sc.get("financial_impact")

    if value is None:
        return 0.0

    dollars = _safe_float(value, 0.0)
    if dollars <= 250_000:
        return 0.2
    if dollars <= 1_000_000:
        return 0.5
    if dollars <= 5_000_000:
        return 0.8
    return 1.0


# ----------------------------
# Main reasoner
# ----------------------------

class DecisionTreeReasoner:
    """
    Async decision tree reasoning engine.

    Typical flow in your agent:
    - risk_assessment = await risk_engine.assess_company_risk(...)
    - scenarios = await planning_engine._generate_planning_scenarios(...) or scenario_simulator outputs
    - then:
        out = await decision_tree.reason(company, risk_assessment, scenarios, ...)
    - dispatch out.recommended_actions
    """

    def __init__(self) -> None:
        # Approval thresholds (match your PlanningEngine style)
        self.thresholds = {
            "auto": 0.8,         # high confidence + severe => can auto-trigger pre-approved actions
            "human_review": 0.5, # moderate => needs ops/manager review
            "executive": 0.3,    # high $$ / high risk => exec sign-off
        }

        # You can tune these weights based on your rubric / real stakeholders
        self.scoring_weights = {
            "composite_risk": 0.40,
            "service_impact": 0.25,
            "financial_impact": 0.25,
            "operational_issues": 0.10,
        }

    async def reason(
        self,
        company: Any,
        risk_assessment: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        *,
        buffer_policy: Optional[Dict[str, Any]] = None,
        supplier_reallocation: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> DecisionOutcome:
        """
        Entry point.
        Builds the decision tree, evaluates it, and returns traceable decisions.
        """
        ctx = DecisionContext(
            company_id=str(getattr(company, "id", "unknown")),
            timestamp=datetime.utcnow(),
            risk_assessment=risk_assessment or {},
            scenarios=scenarios or [],
            buffer_policy=buffer_policy,
            supplier_reallocation=supplier_reallocation,
            constraints=constraints or {},
        )

        tree = await self._build_tree(ctx)
        outcome = await self._evaluate_tree(tree, ctx)
        return outcome

    # ----------------------------
    # Tree building
    # ----------------------------

    async def _build_tree(self, ctx: DecisionContext) -> DecisionNode:
        """
        Build a practical decision tree for supply chain response.

        Logic overview (simple but real-world):
        1) Check overall composite risk
        2) Check if shipping disruption scenario causes big service impact
        3) Check if supplier disruption scenario causes big financial impact
        4) Choose action bundle + approval requirement
        """

        # Leaf bundles (actions are machine-friendly so your agent can execute them)
        leaf_low = DecisionNode(
            node_id="LEAF_LOW",
            description="Low risk: monitor and minor improvements",
            leaf_approval="auto",
            leaf_score=0.2,
            leaf_actions=[
                {"type": "monitoring", "priority": "low", "task": "increase_monitoring_frequency", "frequency": "weekly"},
                {"type": "inventory", "priority": "low", "task": "review_safety_stock", "scope": "critical_items_only"},
            ],
        )

        leaf_medium = DecisionNode(
            node_id="LEAF_MEDIUM",
            description="Medium risk: implement preventive mitigations",
            leaf_approval="human_review",
            leaf_score=0.55,
            leaf_actions=[
                {"type": "supplier", "priority": "medium", "task": "qualify_backup_suppliers", "count": 2},
                {"type": "inventory", "priority": "medium", "task": "increase_buffer_stock", "multiplier": 1.1},
                {"type": "logistics", "priority": "medium", "task": "prebook_alternate_routes", "modes": ["air", "rail"]},
            ],
        )

        leaf_high_shipping = DecisionNode(
            node_id="LEAF_HIGH_SHIPPING",
            description="High risk driven by shipping: reroute + buffer + customer promise review",
            leaf_approval="executive",
            leaf_score=0.85,
            leaf_actions=[
                {"type": "logistics", "priority": "high", "task": "activate_rerouting_plan", "fallback_routes": 2},
                {"type": "inventory", "priority": "high", "task": "increase_buffer_stock", "multiplier": 1.25},
                {"type": "customer", "priority": "high", "task": "review_service_commitments", "policy": "protect_top_tier"},
            ],
        )

        leaf_high_supplier = DecisionNode(
            node_id="LEAF_HIGH_SUPPLIER",
            description="High risk driven by supplier: reallocate spend + qualify alternates + dual-source",
            leaf_approval="executive",
            leaf_score=0.90,
            leaf_actions=[
                {"type": "supplier_reallocation", "priority": "high", "task": "shift_volume_to_backup", "target_share": 0.30},
                {"type": "supplier", "priority": "high", "task": "dual_source_critical_parts", "target_parts": "top_10"},
                {"type": "inventory", "priority": "high", "task": "build_bridge_stock", "days_cover": 21},
            ],
        )

        leaf_critical = DecisionNode(
            node_id="LEAF_CRITICAL",
            description="Critical risk: crisis mode",
            leaf_approval="executive",
            leaf_score=1.0,
            leaf_actions=[
                {"type": "governance", "priority": "critical", "task": "activate_crisis_war_room", "cadence": "daily"},
                {"type": "supplier_reallocation", "priority": "critical", "task": "emergency_rebalance_supply", "target_share": 0.50},
                {"type": "logistics", "priority": "critical", "task": "expedite_shipments", "capex_limit": "preapproved"},
                {"type": "inventory", "priority": "critical", "task": "maximum_buffer_stock", "multiplier": 1.5},
            ],
        )

        # Predicates
        def is_critical(ctx_: DecisionContext) -> bool:
            score = _safe_float(ctx_.risk_assessment.get("composite_risk_score"), 0.0)
            return score >= 0.8

        def is_high(ctx_: DecisionContext) -> bool:
            score = _safe_float(ctx_.risk_assessment.get("composite_risk_score"), 0.0)
            return 0.6 <= score < 0.8

        def shipping_service_hit(ctx_: DecisionContext) -> bool:
            sc = _pick_scenario(ctx_.scenarios, "shipping")
            return _scenario_service_impact(sc) >= 0.35  # >=35% normalized hit

        def supplier_financial_hit(ctx_: DecisionContext) -> bool:
            sc = _pick_scenario(ctx_.scenarios, "supplier")
            return _scenario_financial_impact(sc) >= 0.8  # large $ hit bucket

        def is_medium(ctx_: DecisionContext) -> bool:
            score = _safe_float(ctx_.risk_assessment.get("composite_risk_score"), 0.0)
            return 0.4 <= score < 0.6

        # Build the tree
        # Root: critical?
        root = DecisionNode(
            node_id="N0",
            description="Is composite risk critical (>=0.8)?",
            predicate=is_critical,
            true_child=leaf_critical,
            false_child=None,
        )

        # If not critical: high?
        n1 = DecisionNode(
            node_id="N1",
            description="Is composite risk high (0.6-0.8)?",
            predicate=is_high,
            true_child=None,
            false_child=None,
        )
        root.false_child = n1

        # For high: decide primary driver (shipping vs supplier)
        n2 = DecisionNode(
            node_id="N2",
            description="In high risk: is shipping disruption causing big service hit?",
            predicate=shipping_service_hit,
            true_child=leaf_high_shipping,
            false_child=None,
        )
        n1.true_child = n2

        n3 = DecisionNode(
            node_id="N3",
            description="If not shipping-driven: is supplier disruption causing big financial hit?",
            predicate=supplier_financial_hit,
            true_child=leaf_high_supplier,
            false_child=leaf_medium,  # still high risk, but not strongly attributed -> medium-like bundle
        )
        n2.false_child = n3

        # If not high: medium?
        n4 = DecisionNode(
            node_id="N4",
            description="Is composite risk medium (0.4-0.6)?",
            predicate=is_medium,
            true_child=leaf_medium,
            false_child=leaf_low,
        )
        n1.false_child = n4

        return root

    # ----------------------------
    # Tree evaluation
    # ----------------------------

    async def _evaluate_tree(self, root: DecisionNode, ctx: DecisionContext) -> DecisionOutcome:
        """
        Walks the tree and collects a trace.
        Then computes a decision score + approval + confidence.
        """
        trace: List[Dict[str, Any]] = []
        node = root

        while not node.is_leaf():
            assert node.predicate is not None
            result = bool(node.predicate(ctx))
            trace.append({
                "node_id": node.node_id,
                "description": node.description,
                "result": result,
            })
            node = node.true_child if result else node.false_child
            if node is None:
                # Safety fallback: if the tree is miswired, default to low
                logger.warning("Decision tree missing child; falling back to safe default.")
                return DecisionOutcome(
                    approval_level="human_review",
                    recommended_actions=[{"type": "monitoring", "priority": "medium", "task": "manual_review_required"}],
                    decision_trace=trace,
                    decision_score=0.5,
                    confidence=0.4,
                    metadata={"fallback": True},
                )

        # Leaf reached
        leaf_actions = node.leaf_actions or []
        leaf_approval = node.leaf_approval or "human_review"
        leaf_score = _clamp(_safe_float(node.leaf_score, 0.5))

        trace.append({
            "node_id": node.node_id,
            "description": node.description,
            "leaf": True,
            "approval": leaf_approval,
            "leaf_score": leaf_score,
            "actions_count": len(leaf_actions),
        })

        # Compute a numeric decision_score from signals (useful for dashboards / ranking)
        decision_score = self._compute_decision_score(ctx, leaf_score)

        # Determine approval level (your agent can route to different handlers)
        approval = self._compute_approval(decision_score, ctx, leaf_approval)

        # Confidence: based on how much data we have + consistency
        confidence = self._compute_confidence(ctx, approval, trace)

        # If user already computed supplier_reallocation / buffer_policy,
        # we can attach them as action payloads for execution.
        enriched_actions = self._enrich_actions(leaf_actions, ctx)

        return DecisionOutcome(
            approval_level=approval,
            recommended_actions=enriched_actions,
            decision_trace=trace,
            decision_score=decision_score,
            confidence=confidence,
            metadata={
                "composite_risk_score": _safe_float(ctx.risk_assessment.get("composite_risk_score"), 0.0),
                "risk_level": _risk_level(_safe_float(ctx.risk_assessment.get("composite_risk_score"), 0.0)),
            },
        )

    def _compute_decision_score(self, ctx: DecisionContext, leaf_score: float) -> float:
        """
        Combine leaf severity with real signals into a 0..1 score.
        """
        risk = _clamp(_safe_float(ctx.risk_assessment.get("composite_risk_score"), 0.0))

        # operational issues signal (if present)
        operational_issues = 0.0
        op_issues = ctx.risk_assessment.get("operational_issues")
        if isinstance(op_issues, (int, float)):
            operational_issues = _clamp(float(op_issues) / 10.0)  # normalize: 10 issues ~ 1.0

        # scenario signals
        ship = _pick_scenario(ctx.scenarios, "shipping")
        supp = _pick_scenario(ctx.scenarios, "supplier")

        service_impact = max(_scenario_service_impact(ship), _scenario_service_impact(supp))
        fin_impact = max(_scenario_financial_impact(ship), _scenario_financial_impact(supp))

        base = (
            risk * self.scoring_weights["composite_risk"]
            + service_impact * self.scoring_weights["service_impact"]
            + fin_impact * self.scoring_weights["financial_impact"]
            + operational_issues * self.scoring_weights["operational_issues"]
        )

        # Mix with leaf score so the rule-path matters
        score = 0.65 * base + 0.35 * leaf_score
        return _clamp(score)

    def _compute_approval(self, decision_score: float, ctx: DecisionContext, leaf_approval: str) -> str:
        """
        Determine approval gating:
        - exec if score very high OR financial impact huge
        - else human_review for mid
        - else auto
        """
        # hard trigger exec if any scenario has very high financial impact
        supplier_sc = _pick_scenario(ctx.scenarios, "supplier")
        ship_sc = _pick_scenario(ctx.scenarios, "shipping")
        fin = max(_scenario_financial_impact(supplier_sc), _scenario_financial_impact(ship_sc))
        if fin >= 1.0:  # >$5M bucket
            return "executive"

        # Otherwise decide by score
        if decision_score >= self.thresholds["auto"]:
            # still respect leaf_approval if it requires more
            return leaf_approval if leaf_approval in ("human_review", "executive") else "auto"
        if decision_score >= self.thresholds["human_review"]:
            return "human_review"
        return "executive" if leaf_approval == "executive" else "human_review"

    def _compute_confidence(self, ctx: DecisionContext, approval: str, trace: List[Dict[str, Any]]) -> float:
        """
        Confidence heuristic:
        - More scenarios + richer risk components -> higher confidence
        - Executive decisions are often lower confidence unless data is strong
        """
        risk = ctx.risk_assessment or {}
        components = risk.get("risk_components", {})
        has_components = isinstance(components, dict) and len(components) >= 3

        scenario_count = len(ctx.scenarios)
        scenario_score = _clamp(scenario_count / 6.0)  # ~6 scenarios => 1.0

        data_score = 0.5 + (0.25 if has_components else 0.0) + 0.25 * scenario_score

        # penalize if tree was too shallow (little reasoning) or too many missing branches
        depth_score = _clamp(len(trace) / 5.0)

        conf = 0.6 * data_score + 0.4 * depth_score

        if approval == "executive":
            conf *= 0.9  # be slightly conservative
        return _clamp(conf)

    def _enrich_actions(self, actions: List[Dict[str, Any]], ctx: DecisionContext) -> List[Dict[str, Any]]:
        """
        Attach computed payloads (supplier_reallocation / buffer_policy) if available.
        """
        enriched: List[Dict[str, Any]] = []
        for a in actions:
            a2 = dict(a)
            if a2.get("type") == "supplier_reallocation" and ctx.supplier_reallocation:
                a2["payload"] = ctx.supplier_reallocation
            if a2.get("type") == "inventory" and ctx.buffer_policy:
                # attach recommended buffer policy if present
                a2["payload"] = ctx.buffer_policy
            enriched.append(a2)
        return enriched


# ----------------------------
# Minimal integration example
# ----------------------------
# In your PlanningEngine after you build scenarios and strategies:
#
# reasoner = DecisionTreeReasoner()
# decision = await reasoner.reason(company, risk_assessment, scenarios,
#                                  buffer_policy=buffer_stock_output.get("recommended_policy"),
#                                  supplier_reallocation=supplier_realloc_output,
#                                  constraints={"budget": 1_000_000})
# if decision.approval_level == "auto":
#     # execute decision.recommended_actions
# else:
#     # route for human/executive review