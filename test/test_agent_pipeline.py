from __future__ import annotations

import os
import unittest

from agent.tools import (
    analyze_custom_profile,
    build_business_impact_report,
    build_judging_scorecard,
    build_responsible_ai_report,
    generate_roi_benchmark_report,
    derive_memory_feedback,
    generate_actions,
    get_company_profile,
    log_mock_workflow_execution,
    onboard_company_profile,
    run_cost_optimized_pipeline,
    run_board_demo,
    run_evaluation_harness,
    simulate_what_if_scenarios,
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

    def test_business_impact_and_scorecard(self) -> None:
        company = get_company_profile("de_semiconductor_auto")
        pipeline = run_cost_optimized_pipeline(company)
        risk = score_risk(company, pipeline["events_for_risk"])
        plan = simulate_tradeoffs(company, risk)
        actions = generate_actions(company, risk, plan)
        cvr = {
            "estimated_revenue_saved_usd": 100000.0,
            "estimated_revenue_at_risk_usd": 300000.0,
            "estimated_call_reduction_pct": 50.0,
        }
        impact = build_business_impact_report(company, risk, plan, actions, cvr)
        payload = {
            "pipeline_stats": pipeline["pipeline_stats"],
            "actions": actions,
            "transparency_trace": {"stage_sequence": ["perception"]},
            "business_impact_report": impact,
            "responsible_ai_report": {"status": "pass", "checks": [1, 2, 3]},
            "cost_value_report": cvr,
            "events": pipeline["events_for_risk"],
            "company": company,
            "plan": plan,
            "memory_write": {"saved": True},
        }
        scorecard = build_judging_scorecard(payload)
        self.assertIn("total_score_out_of_100", scorecard)
        self.assertGreaterEqual(scorecard["total_score_out_of_100"], 0)

    def test_board_demo_runs(self) -> None:
        out = run_board_demo()
        self.assertIn("headline", out)
        self.assertIn("best_score_out_of_100", out["headline"])

    def test_what_if_and_onboarding_and_benchmark(self) -> None:
        sim = simulate_what_if_scenarios(
            company_id="de_semiconductor_auto",
            fuel_multiplier=1.15,
            lead_time_shock_days=4,
            demand_shock_pct=8.0,
        )
        self.assertIn("scenario_risk_score", sim)

        onboard = onboard_company_profile(
            company_name="Acme Motion",
            region="Canada",
            industry="industrial_components",
            critical_components_csv="controllers,bearings",
            risk_appetite="low",
        )
        self.assertIn("profile", onboard)
        self.assertEqual(onboard["profile"]["company_name"], "Acme Motion")

        eval_out = run_evaluation_harness()
        self.assertIn("avg_score_out_of_100", eval_out)

        bench = generate_roi_benchmark_report("de_semiconductor_auto")
        self.assertIn("companies", bench)
        self.assertGreaterEqual(len(bench["companies"]), 1)


if __name__ == "__main__":
    unittest.main()
