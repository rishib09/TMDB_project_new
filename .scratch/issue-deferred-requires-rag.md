# Deferred: `requires_rag` is a derivable, redundant field — derive or harden?

## Status

**Deferred by decision** (Option C): keep the field as-is for now; revisit when the LangGraph StateGraph lands. Nothing to implement in #3.

## Background

`requires_rag` is a required field on `QueryRoutingDecision` (added in Issue #1). An audit during Issue #3 found:

### Current usage

| Location | Role |
|---|---|
| `src/domain/routing.py` | Defined (required, LLM-declared via structured output) |
| `src/maya/router.py` system prompt rule 4 | Instructed: "set false for GREETING, CAPABILITIES, OUT_OF_SCOPE" |
| `src/maya/router.py` `_heuristic_fallback` | Set: `requires_rag = (intent == SEMANTIC_SEARCH)` |
| `src/maya/router.py` `_apply_session_exclusions` | **Only production consumer** — gates exclusion-merging |

### The two findings

1. **The headline consumer doesn't exist yet.** The field's real purpose is the LangGraph conditional branch (Route Query Node → retrieval vs. direct response), which is Issue #5 scope. Until then it is nearly decorative.
2. **It is derivable from `intent`**, making it redundant LLM-declared state:

   ```python
   RETRIEVAL_INTENTS = {SEMANTIC_SEARCH, ATTRIBUTE_FILTER, SUPERLATIVE_RANKING, NEGATION_EXCLUSION}
   requires_rag == intent in RETRIEVAL_INTENTS  # per router prompt rule 4
   ```

   Inconsistency spotted: the heuristic fallback derives it *stricter* (`intent == SEMANTIC_SEARCH`) than prompt rule 4 implies — an `ATTRIBUTE_FILTER` would get `False` from the heuristic but `True` from the LLM. Two derivations of the same concept already disagree.

### Redundancy cuts both ways

- **Risk:** LLM can emit contradictory output (`intent=OUT_OF_SCOPE` + `requires_rag=True`) with no guard — we'd trust the wrong one silently.
- **Value:** the disagreement itself is a *diagnostic signal* that the router model is confused — directly relevant to the low-confidence policy analysis in #12.

## Options (when revisiting)

- **Option A — Derive, don't ask (leanest):** Remove from schema; compute from a module-level `RETRIEVAL_INTENTS` constant wherever needed. Single source of truth. Cost: changes the Issue #1 schema + router prompt, small migration.
- **Option B — Keep, but harden:** In `route()`, derive the truth from intent; if the LLM's `requires_rag` contradicts it, trust the intent and override the field. Log/emit mismatches (Langfuse event) as a confusion signal feeding #12.
- **Option C — Keep as-is (current):** Defer until #5 forces the branch consumer decision. ✅ chosen for now.

## When to revisit

When implementing #5 (LangGraph StateGraph), the conditional edge forces this decision:

- If the branch keys off `intent` (recommended), drop or harden `requires_rag` in the same PR.
- Feed any collected mismatch telemetry into #12 (low-confidence policy grilling).

## Dependencies

- **Relates to:** #5 (StateGraph conditional edges — the decision point), #12 (low-confidence grilling — mismatch as confidence signal)
- **Schema origin:** #1 (domain models, closed)

## Non-goals

- No prompt or schema changes until the #5 decision point.
- No behavior change to the exclusion-merge gate (correct today under both derivations for all intents the heuristic can emit).
