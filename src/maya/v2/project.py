"""The v2 -> v1 projection: one seam where old consumers read Understanding.

Everything downstream of routing (retrieve, synthesize, the UI chip, the
eval scorer) keeps reading ``QueryRoutingDecision``. V2 readings enter
that world ONLY through this function — outside the v1 files, per #83
S1–S10. Mood/audience come from the post-dispose preferences snapshot
(the authoritative merge), not the raw delta.
"""

from __future__ import annotations

from src.domain.memory import UserSessionPreferences
from src.domain.routing import (
    IntentType,
    QueryRoutingDecision,
)
from src.maya.v2.models import Understanding

_RETRIEVAL_INTENTS = {
    IntentType.SEMANTIC_SEARCH,
    IntentType.ATTRIBUTE_FILTER,
    IntentType.SUPERLATIVE_RANKING,
    IntentType.NEGATION_EXCLUSION,
}


def project_understanding(
    u: Understanding, prefs: UserSessionPreferences
) -> QueryRoutingDecision:
    """Map an Understanding onto the v1 decision shape (guardrails unchanged).

    ``requires_rag`` is DERIVED from intent, never trusted (v1 normalize
    rule 1); ``confidence`` rides through as telemetry (C11); the decision
    is never a fallback — degradation already happened upstream (C12).
    """
    return QueryRoutingDecision(
        intent=u.intent,
        confidence=u.confidence,
        standalone_query=u.standalone_query,
        requires_rag=u.intent in _RETRIEVAL_INTENTS,
        filters=u.filters,
        mood=prefs.preferred_mood or "",
        audience=prefs.audience or "",
        is_fallback=False,
        reasoning="v2 understanding",  # telemetry: marks the projected origin
    )
