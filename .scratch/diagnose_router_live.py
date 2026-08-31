"""Diagnostic: capture actual live router decisions for the 6 failing queries.

Run via: npx @dotenvx/dotenvx run -- python .scratch/diagnose_router_live.py
Do NOT call load_dotenv() here — the .env is dotenvx-encrypted; dotenvx run
injects the decrypted values and load_dotenv would overwrite them with ciphertext.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState, UserSessionPreferences
from src.maya.router import MayaRouter

if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("OPENROUTER_API_KEY not available")

router = MayaRouter(ExperimentConfig())

QUERIES = [
    ("semantic_search", "movies about space exploration and lunar colonies", ConversationState()),
    ("attribute_filter", "1980s horror movies directed by John Carpenter starring Kurt Russell", ConversationState()),
    ("superlative", "What was the highest-grossing film of 1970?", ConversationState()),
    ("negation", "action movies without Tom Cruise", ConversationState()),
    ("pre_1970", "what are the best movies from the 1950s?", ConversationState()),
    ("exclusions", "recommend a mystery thriller", None),  # state with Horror exclusion
]

for name, query, state in QUERIES:
    if state is None:
        state = ConversationState()
        state.session_preferences = UserSessionPreferences(excluded_genres=["Horror"])
    d = router.route(query, state)
    print(json.dumps({
        "case": name,
        "intent": d.intent.value,
        "confidence": d.confidence,
        "requires_rag": d.requires_rag,
        "standalone_query": d.standalone_query,
        "filters": d.filters.model_dump() if d.filters else None,
        "superlative": d.superlative.model_dump() if d.superlative else None,
        "reasoning": d.reasoning[:120],
    }, indent=1))
