import logging

from .config import Settings
from .schemas import AgentPingResponse, AgentStatus

logger = logging.getLogger(__name__)


class NebiusAgent:
    """Small read-only client for the optional Nebius model health check."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def status(self) -> AgentStatus:
        """Return configuration status without exposing the API key."""
        configured = bool(self.settings.nebius_api_key)
        return AgentStatus(
            base_url=self.settings.nebius_base_url,
            configured=configured,
            primary_model=self.settings.nebius_model,
            fallback_model=self.settings.nebius_fallback_model,
            message="API key configured." if configured else "NEBIUS_API_KEY is missing; agent calls are disabled.",
        )

    async def ping(self) -> AgentPingResponse:
        """Probe the primary model, then the fallback, and always close the client."""
        if not self.settings.nebius_api_key:
            return AgentPingResponse(
                status="not_configured",
                message="Set NEBIUS_API_KEY to enable the Nebius GLM ping.",
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return AgentPingResponse(
                status="unavailable",
                message="Install backend requirements to enable the Nebius GLM client.",
            )

        client = AsyncOpenAI(
            api_key=self.settings.nebius_api_key,
            base_url=self.settings.nebius_base_url,
        )
        try:
            try:
                result = await client.chat.completions.create(
                    model=self.settings.nebius_model,
                    messages=[{"role": "user", "content": "Reply with exactly: swatterfly phase 1 ready"}],
                    max_tokens=32,
                    temperature=0,
                )
                return AgentPingResponse(
                    status="ok",
                    model=self.settings.nebius_model,
                    response=result.choices[0].message.content or "",
                    message="Primary model responded.",
                )
            except Exception:
                logger.warning("Nebius primary model unavailable; trying fallback", exc_info=False)
                try:
                    result = await client.chat.completions.create(
                        model=self.settings.nebius_fallback_model,
                        messages=[{"role": "user", "content": "Reply with exactly: swatterfly phase 1 ready"}],
                        max_tokens=32,
                        temperature=0,
                    )
                    return AgentPingResponse(
                        status="ok",
                        model=self.settings.nebius_fallback_model,
                        response=result.choices[0].message.content or "",
                        message="Fallback model responded.",
                    )
                except Exception:
                    logger.warning("Nebius primary and fallback models unavailable", exc_info=False)
                    return AgentPingResponse(
                        status="error",
                        message="Nebius GLM ping failed for both configured models; no key or response was logged.",
                    )
        finally:
            await client.close()
