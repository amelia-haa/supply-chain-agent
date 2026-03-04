from __future__ import annotations

import os
import unittest

from agent.tools import (
    analyze_custom_profile,
    build_responsible_ai_report,
    derive_memory_feedback,
    generate_actions,
    get_company_profile,
    log_mock_workflow_execution,
    run_cost_optimized_pipeline,
    score_risk,
    simulate_tradeoffs,
)


class AgentPipelineTests(unittest.TestCase):
    def test_analyze_custom_profile_runs(self) -> None:
        custom = {
            "company_id": "custom_semicon_test",
            "company_name": "Custom Semicon GmbH",
            "region": "Germany",
            "industry": "electronics",
            "risk_appetite": "low",
            "critical_components": ["semiconductors", "controllers"],
            "supplier_concentration": {
                "semiconductors": {"top_supplier_share": 0.84, "region": "East Asia"}
            },
            "inventory_policy": {"semiconductors_days_buffer": 6},
            "lead_time_sensitivity": "high",
        }
        result = analyze_custom_profile(custom)
        self.assertIn("risk", result)
        self.assertIn("actions", result)
        self.assertEqual(result["company"]["company_id"], "custom_semicon_test")

    def test_personalized_plans_differ_between_companies(self) -> None:
        de = get_company_profile("de_semiconductor_auto")
        mx = get_company_profile("mx_multisource_industrial")
        risk_stub = {"risk_level": "high", "risk_score": 0.76}

        de_plan = simulate_tradeoffs(de, risk_stub)
        mx_plan = simulate_tradeoffs(mx, risk_stub)

        self.assertNotEqual(de_plan[0]["action"], mx_plan[0]["action"])

    def test_critical_profile_increases_risk(self) -> None:
        prev = os.environ.get("APP_SIGNAL_PROFILE")
        os.environ["APP_SIGNAL_PROFILE"] = "critical"
        try:
            company = get_company_profile("de_semiconductor_auto")
            pipeline = run_cost_optimized_pipeline(company)
            risk = score_risk(company, pipeline["events_for_risk"])
            self.assertGreaterEqual(risk["risk_score"], 0.5)
        finally:
            if prev is None:
                os.environ.pop("APP_SIGNAL_PROFILE", None)
            else:
                os.environ["APP_SIGNAL_PROFILE"] = prev

    def test_actions_include_guardrails_and_tier(self) -> None:
        company = get_company_profile("de_semiconductor_auto")
        risk = {"risk_score": 0.71, "risk_level": "high", "reasons": ["x"]}
        plan = [
            {"action": "Very expensive plan", "cost_usd": 500000, "service_gain": 0.2, "resilience_gain": 0.3},
            {"action": "Safer budget plan", "cost_usd": 120000, "service_gain": 0.15, "resilience_gain": 0.2},
        ]
        actions = generate_actions(company, risk, plan)
        self.assertIn(actions["tiered_alert_action"], {"ai_mitigation_planning", "executive_escalation", "dashboard_alert", "store_signal_only"})
        self.assertIn("guardrail_flags", actions)
        self.assertEqual(actions["recommended_top_action"]["action"], "Safer budget plan")

    def test_memory_feedback_has_outcome_field(self) -> None:
        feedback = derive_memory_feedback("MidMarket Auto Parts Co")
        self.assertIn("avg_mitigation_success", feedback)

    def test_responsible_ai_report_shape(self) -> None:
        company = get_company_profile("de_semiconductor_auto")
        pipeline = run_cost_optimized_pipeline(company)
        risk = score_risk(company, pipeline["events_for_risk"])
        plan = simulate_tradeoffs(company, risk)
        actions = generate_actions(company, risk, plan)
        report = build_responsible_ai_report(company, risk, plan, pipeline["events_for_risk"], actions)
        self.assertIn(report["status"], {"pass", "warning", "fail"})
        self.assertIn("checks", report)
        self.assertIn("override_policy", report)

    def test_workflow_log_returns_integration_results(self) -> None:
        company = get_company_profile("de_semiconductor_auto")
        risk = {"risk_score": 0.42, "risk_level": "low", "reasons": []}
        actions = {"triggered_workflows": [], "guardrail_flags": []}
        out = log_mock_workflow_execution(company, risk, actions)
        self.assertIn("integration_results", out)
        self.assertGreaterEqual(len(out["integration_results"]), 1)


if __name__ == "__main__":
    unittest.main()
