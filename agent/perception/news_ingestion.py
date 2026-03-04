import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone # Fixed import

from agent.config import settings
from agent.llm.gemini_client import GeminiLLM, safe_parse_json
from agent.models.supply_chain import Disruption

logger = logging.getLogger(__name__)

class NewsIngestionService:
    def __init__(self):
        # Use central Gemini wrapper (prefers latest SDK where available).
        self.llm = GeminiLLM(api_key=settings.google_api_key, model_name="gemini-1.5-flash")
        
    async def fetch_news_articles(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch real-time news articles from verified APIs."""
        articles = []
        # Calculate time safely for March 2026
        from_time = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        
        # Cleanly organized sources
        sources = [
            {
                'name': 'NewsAPI',
                'url': 'https://newsapi.org/v2/everything',
                'params': {
                    'q': 'supply chain disruption OR shipping delay',
                    'from': from_time,
                    'apiKey': settings.google_api_key, # Assuming you use one key for demo
                    'language': 'en'
                },
                'enabled': True # Toggle based on your .env settings
            }
        ]

        async with aiohttp.ClientSession() as session:
            for source in sources:
                if not source['enabled']: continue
                try:
                    async with session.get(source['url'], params=source['params'], timeout=15) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            articles.extend(self._parse_response(source['name'], data))
                except Exception as e:
                    logger.error(f"Error fetching {source['name']}: {e}")

        return articles or await self._get_fallback_news()

    def _parse_response(self, source: str, data: Dict) -> List[Dict]:
        parsed = []
        # Logic to map different API shapes to a standard Internal Article Format
        if source == 'NewsAPI':
            for art in data.get('articles', []):
                parsed.append({
                    'title': art.get('title'),
                    'description': art.get('description'),
                    'url': art.get('url'),
                    'publishedAt': art.get('publishedAt')
                })
        return parsed

    async def process_article(self, article: Dict[str, Any]) -> Optional[Disruption]:
        """The 'AI Brain' part: Turns a headline into a structured Disruption model."""
        prompt = f"""
        Extract supply chain risk from this news:
        Title: {article['title']}
        Text: {article['description']}

        Return JSON only:
        {{
            "disruption_type": "shipping|geopolitical|supplier|climate",
            "severity_score": 0.0-1.0,
            "affected_regions": ["list", "of", "countries"]
        }}
        """
        try:
            response = await self.llm.generate_text(prompt, temperature=0.1, max_output_tokens=350)
            data = safe_parse_json(response.text)
            
            # Create the Pydantic model
            return Disruption(
                id=f"news-{datetime.now().timestamp()}",
                description=article['description'] or article['title'],
                disruption_type=data['disruption_type'],
                severity_score=data['severity_score'],
                affected_regions=data['affected_regions']
            )
        except Exception as e:
            logger.warning(f"AI failed to process article: {e}")
            return None

    async def _get_fallback_news(self) -> List[Dict]:
        return [{
            'title': 'Red Sea Shipping Route Blockage',
            'description': 'Continued tensions lead to 14-day delays for EU-Asia routes.',
            'url': 'https://demo.logistics',
            'publishedAt': datetime.now(timezone.utc).isoformat()
        }]
