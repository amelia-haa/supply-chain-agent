# src/llm/gemini_client.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    raw: Optional[Any] = None


class GeminiLLM:
    """
    Async-safe Gemini wrapper.
    - Tries "google.generativeai" (older style) first if installed/configured.
    - Falls back to "google.genai" client style if installed.
    - Always exposes: await generate_text(prompt, **params) -> LLMResult
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-pro",
        timeout_s: float = 25.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        self._mode = None
        self._model = None
        self._client = None

        # Try older sdk: google.generativeai
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._mode = "google.generativeai"
            logger.info("GeminiLLM using google.generativeai (%s)", self.model_name)
            return
        except Exception as e:
            logger.warning("google.generativeai not available/failed init: %s", e)

        # Try newer sdk: google.genai
        try:
            from google import genai  # type: ignore

            # New docs commonly use genai.Client() :contentReference[oaicite:1]{index=1}
            # Some envs use env var GOOGLE_API_KEY; we set it for safety.
            os.environ.setdefault("GOOGLE_API_KEY", self.api_key)
            self._client = genai.Client(api_key=self.api_key)  # works even if env var isn't set
            self._mode = "google.genai"
            logger.info("GeminiLLM using google.genai (%s)", self.model_name)
            return
        except Exception as e:
            raise RuntimeError(
                "No working Gemini SDK found. Install google-generativeai OR google-genai."
            ) from e

    async def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 900,
    ) -> LLMResult:
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self._mode == "google.generativeai":
                    # Some versions have generate_content_async; if not, run sync in a thread.
                    if hasattr(self._model, "generate_content_async"):
                        coro = self._model.generate_content_async(  # type: ignore[attr-defined]
                            prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": max_output_tokens,
                            },
                        )
                        resp = await asyncio.wait_for(coro, timeout=self.timeout_s)
                        return LLMResult(text=(resp.text or "").strip(), raw=resp)

                    # Sync fallback -> thread
                    def _sync_call() -> Any:
                        return self._model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": max_output_tokens,
                            },
                        )

                    resp = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=self.timeout_s)
                    return LLMResult(text=(resp.text or "").strip(), raw=resp)

                if self._mode == "google.genai":
                    # New client style -> usually sync; run in a thread to keep your agent async.
                    def _sync_call() -> Any:
                        return self._client.models.generate_content(  # type: ignore[union-attr]
                            model=self.model_name,
                            contents=prompt,
                            config={
                                "temperature": temperature,
                                "max_output_tokens": max_output_tokens,
                            },
                        )

                    resp = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=self.timeout_s)

                    # Different SDKs expose text differently; try common fields:
                    text = ""
                    if hasattr(resp, "text"):
                        text = (resp.text or "").strip()
                    else:
                        text = str(resp).strip()

                    return LLMResult(text=text, raw=resp)

                raise RuntimeError("GeminiLLM is not initialized correctly.")

            except Exception as e:
                last_err = e
                backoff = 0.6 * (2 ** attempt)
                logger.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, self.max_retries + 1, e)
                await asyncio.sleep(min(backoff, 3.0))

        raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Attempts to parse JSON even if the model wraps it in extra text.
    """
    text = text.strip()

    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Could not parse JSON from LLM output.")