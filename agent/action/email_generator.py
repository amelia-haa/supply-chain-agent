# src/autonomous_action/email_generator.py
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

from agent.config import settings
from agent.models import Company, Supplier, MitigationAction
from agent.llm.gemini_client import GeminiLLM, safe_parse_json

logger = logging.getLogger(__name__)


class EmailGenerator:
    def __init__(self):
        # You can change model_name to whatever you’re using.
        # (Newer docs show "gemini-3-..." examples, but your project can keep a stable model.)
        self.llm = GeminiLLM(
            api_key=settings.google_api_key,
            model_name=getattr(settings, "gemini_model", "gemini-1.5-pro"),
            timeout_s=25.0,
            max_retries=2,
        )

        self.email_templates = {
            "supplier_outreach": {
                "subject_template": "Partnership Opportunity - {company_name}",
                "tone": "professional",
            },
            "negotiation": {
                "subject_template": "Contract Review Discussion - {supplier_name}",
                "tone": "collaborative",
            },
            "escalation": {
                "subject_template": "URGENT: Supply Chain Risk Mitigation Required",
                "tone": "urgent",
            },
            "inventory_notification": {
                "subject_template": "Inventory Level Adjustment Notice",
                "tone": "informative",
            },
            "logistics_notification": {
                "subject_template": "Shipping Route Update - {route_change}",
                "tone": "operational",
            },
        }

    # ---------- Public APIs (agent calls these) ----------

    async def generate_supplier_outreach_emails(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        potential_suppliers = [
            {
                "name": "Global Components Ltd",
                "location": "Vietnam",
                "specialization": "Electronic components",
                "contact_person": "John Smith",
                "email": "john.smith@globalcomponents.com",
            },
            {
                "name": "Asia Pacific Manufacturing",
                "location": "Thailand",
                "specialization": "Automotive parts",
                "contact_person": "Sarah Chen",
                "email": "sarah.chen@apmanufacturing.com",
            },
            {
                "name": "European Supply Solutions",
                "location": "Poland",
                "specialization": "Industrial components",
                "contact_person": "Marco Rossi",
                "email": "marco.rossi@eurosupply.com",
            },
        ]

        emails = []
        for s in potential_suppliers:
            emails.append(await self._create_supplier_outreach_email(company, s, action))

        return {
            "email_type": "supplier_outreach",
            "total_emails": len(emails),
            "emails": emails,
            "generated_at": datetime.utcnow(),
            "next_follow_up": datetime.utcnow() + timedelta(days=7),
        }

    async def generate_negotiation_emails(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        existing_suppliers = [
            {
                "name": "Current Supplier A",
                "contact_person": "David Wilson",
                "email": "david.wilson@suppliera.com",
                "contract_end_date": "2026-12-31",
                "current_terms": "Standard 30-day payment terms",
            },
            {
                "name": "Current Supplier B",
                "contact_person": "Lisa Zhang",
                "email": "lisa.zhang@supplierb.com",
                "contract_end_date": "2026-09-30",
                "current_terms": "Standard 45-day payment terms",
            },
        ]

        emails = []
        for s in existing_suppliers:
            emails.append(await self._create_negotiation_email(company, s, action))

        return {
            "email_type": "negotiation",
            "total_emails": len(emails),
            "emails": emails,
            "generated_at": datetime.utcnow(),
            "negotiation_deadline": datetime.utcnow() + timedelta(days=30),
        }

    async def generate_escalation_notifications(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        recipients = [
            {"name": "Supply Chain Director", "email": "sc.director@company.com", "role": "executive"},
            {"name": "Operations Manager", "email": "ops.manager@company.com", "role": "management"},
            {"name": "Risk Management Team", "email": "risk.team@company.com", "role": "team"},
        ]

        emails = []
        for r in recipients:
            emails.append(await self._create_escalation_email(company, r, action))

        return {
            "email_type": "escalation",
            "total_emails": len(emails),
            "emails": emails,
            "generated_at": datetime.utcnow(),
            "escalation_level": action.priority_level,
            "response_required_by": datetime.utcnow() + timedelta(days=1),
        }

    # ---------- LLM-backed builders (private) ----------

    async def _create_supplier_outreach_email(self, company: Company, supplier: Dict, action: MitigationAction) -> Dict[str, Any]:
        template = self.email_templates["supplier_outreach"]
        subject = template["subject_template"].format(company_name=company.name)

        payload = {
            "email_type": "supplier_outreach",
            "tone": template["tone"],
            "company": {"name": company.name, "industry": company.industry, "location": company.location},
            "recipient": {
                "name": supplier["contact_person"],
                "email": supplier["email"],
                "supplier_name": supplier["name"],
                "supplier_location": supplier["location"],
                "specialization": supplier["specialization"],
            },
            "context": {
                "purpose": "Exploring partnership opportunities for supply chain diversification",
                "action_title": action.title,
                "action_description": action.description,
            },
            "constraints": {
                "max_words": 220,
                "must_include": ["intro", "brief company overview", "why reaching out", "clear next steps", "signature"],
            },
        }

        return await self._generate_email_json(
            subject=subject,
            payload=payload,
            fallback=self._fallback_email(
                recipient_name=supplier["contact_person"],
                recipient_email=supplier["email"],
                subject=subject,
                body=(
                    f"Dear {supplier['contact_person']},\n\n"
                    f"My name is {company.name}'s Supply Chain team. We are exploring additional suppliers to strengthen "
                    f"resilience and would like to learn more about {supplier['name']}'s capabilities in {supplier['specialization']}.\n\n"
                    f"If you’re open to a short introductory call next week, please share availability and a capability overview "
                    f"(product lines, lead times, certifications, and MOQ).\n\n"
                    f"Best regards,\n{company.name} Supply Chain"
                ),
                email_type="supplier_outreach",
                priority="medium",
                supplier_name=supplier["name"],
            ),
        )

    async def _create_negotiation_email(self, company: Company, supplier: Dict, action: MitigationAction) -> Dict[str, Any]:
        template = self.email_templates["negotiation"]
        subject = template["subject_template"].format(supplier_name=supplier["name"])

        payload = {
            "email_type": "negotiation",
            "tone": template["tone"],
            "company": {"name": company.name, "industry": company.industry},
            "recipient": {
                "name": supplier["contact_person"],
                "email": supplier["email"],
                "supplier_name": supplier["name"],
            },
            "context": {
                "contract_end_date": supplier["contract_end_date"],
                "current_terms": supplier["current_terms"],
                "goal": "Schedule a contract review to add flexibility & risk-sharing clauses",
                "action_title": action.title,
            },
            "constraints": {"max_words": 220, "include_bullets_for_topics": True},
        }

        return await self._generate_email_json(
            subject=subject,
            payload=payload,
            fallback=self._fallback_email(
                recipient_name=supplier["contact_person"],
                recipient_email=supplier["email"],
                subject=subject,
                body=(
                    f"Dear {supplier['contact_person']},\n\n"
                    f"Thank you for your continued partnership. With our contract renewal approaching "
                    f"({supplier['contract_end_date']}), we’d like to schedule a review to discuss improvements that benefit both teams.\n\n"
                    f"Topics:\n- Flexibility clauses for lead times and allocation\n- Communication and escalation pathways\n- Service-level and continuity expectations\n\n"
                    f"Could we set up a 30-minute call in the next two weeks?\n\n"
                    f"Best regards,\n{company.name} Procurement"
                ),
                email_type="negotiation",
                priority="high",
                supplier_name=supplier["name"],
            ),
        )

    async def _create_escalation_email(self, company: Company, recipient: Dict, action: MitigationAction) -> Dict[str, Any]:
        template = self.email_templates["escalation"]
        subject = template["subject_template"]

        payload = {
            "email_type": "escalation",
            "tone": template["tone"],
            "company": {"name": company.name, "industry": company.industry},
            "recipient": recipient,
            "context": {
                "issue": action.title,
                "details": action.description,
                "priority_level": action.priority_level,
                "urgency_score": getattr(action, "urgency_score", 0.7),
                "deadline_hours": 24,
            },
            "constraints": {"max_words": 190, "must_include": ["impact", "decision_needed", "deadline", "owner"]},
        }

        return await self._generate_email_json(
            subject=subject,
            payload=payload,
            fallback=self._fallback_email(
                recipient_name=recipient["name"],
                recipient_email=recipient["email"],
                subject=subject,
                body=(
                    f"Dear {recipient['name']},\n\n"
                    f"URGENT: {action.title}\n\n"
                    f"Summary: {action.description}\n"
                    f"Priority: {action.priority_level}\n\n"
                    f"Decision needed within 24 hours: approve immediate mitigation actions and allocate resources.\n\n"
                    f"Regards,\nSupply Chain Risk"
                ),
                email_type="escalation",
                priority="critical",
                supplier_name=None,
            ),
        )

    async def _generate_email_json(self, *, subject: str, payload: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls the LLM and forces strict JSON.
        This prevents the “LLM wrote random formatting” problem.
        """
        system_rules = (
            "Return ONLY valid JSON.\n"
            "JSON schema:\n"
            "{\n"
            '  "subject": string,\n'
            '  "body": string,\n'
            '  "salutation": string,\n'
            '  "closing": string,\n'
            '  "key_points": [string, ...]\n'
            "}\n"
            "No markdown. No extra commentary."
        )

        prompt = f"{system_rules}\n\nINPUT:\n{payload}\n\nUse the provided subject exactly:\n{subject}"

        try:
            result = await self.llm.generate_text(prompt, temperature=0.3, max_output_tokens=900)
            data = safe_parse_json(result.text)

            body = str(data.get("body", "")).strip()
            out_subject = str(data.get("subject", subject)).strip() or subject

            if not body:
                raise ValueError("LLM returned empty body.")

            enriched = dict(fallback)
            enriched.update(
                {
                    "subject": out_subject,
                    "body": body,
                    "generated_at": datetime.utcnow(),
                    "fallback_used": False,
                    "key_points": data.get("key_points", []),
                }
            )
            return enriched

        except Exception as e:
            logger.error("LLM email generation failed, using fallback. Error=%s", e)
            fallback["generated_at"] = datetime.utcnow()
            fallback["fallback_used"] = True
            fallback["error"] = str(e)
            return fallback

    def _fallback_email(
        self,
        *,
        recipient_name: str,
        recipient_email: str,
        subject: str,
        body: str,
        email_type: str,
        priority: str,
        supplier_name: str | None,
    ) -> Dict[str, Any]:
        return {
            "recipient_name": recipient_name,
            "recipient_email": recipient_email,
            "supplier_name": supplier_name,
            "subject": subject,
            "body": body,
            "type": email_type,
            "priority": priority,
            "generated_at": datetime.utcnow(),
        }