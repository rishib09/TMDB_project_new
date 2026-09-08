# ADR 0009: Recall replays input, never output — no query/response caching

Date: 2026-09-08
Status: Accepted
Issues: #28 (grill), #48 (implementation)

## Context

Issue #28 proposed two features under "query cache": history recall in the
chat UI and a latency cache (exact or semantic) over parts of the pipeline.
Maya's answers are state-dependent: Persistent Exclusions, funnel state,
mood/audience axes, and fresh-start resets all legitimately change the
result of the same utterance. A query→output cache at any layer must
therefore key on Conversation State, or it is correctness-breaking.

Facts established during the grill:

- No per-stage wall-clock latencies are measured anywhere (router,
  retrieval, synthesis) — a latency cache has no evidence of a problem to
  solve ("measure before building" is unmet).
- Queries flow nearly raw (`.strip()` only); an exact-match key would miss
  trivial variations, and a similarity key adds a threshold knob plus a
  false-positive budget for a local single-user app.
- No cross-session query store exists: `turn_log` is in-memory, the trace
  ring is in-memory (max 100), `FeedbackStore` persists ratings without
  query text.
- `st.chat_input` cannot be pre-filled programmatically, and requirements
  are frozen — terminal-style up-arrow injection is not implementable.

## Decision

**Recall replays input, never output.** A recalled query is an ordinary
turn: full pipeline, current Conversation State, every guardrail. No cached
router decision, retrieval result, or synthesized response may ride along.

Scope of the implementation (#48):

- Session-only history read from `turn_log` — last 10 routed queries,
  deduplicated, newest first; funnel-owned turns excluded.
- Popover above the chat input with Resend and Edit-then-send; no input-box
  injection.
- `recalled: true` on the turn row and a `recall` node in the local trace
  ring — recall usage is observable (telemetry contract), and it is the
  evidence base for ever revisiting a latency cache.
- No recall-specific guards: token cap, budget, and funnel logic already
  run because the turn is ordinary.

## Rejected

- **Response cache** — correctness-breaking under state-dependence.
- **Retrieval cache** — a correct key must serialize exclusions + funnel +
  mood/audience state; hit rates collapse while invalidation (weekly data
  refresh, index rebuilds) remains unsolved.
- **Router decision cache** — the only plausibly deterministic-safe layer,
  saving one small LLM call; rejected for now as unmeasured benefit. May be
  revisited only with stage-latency measurements and observed recall usage.
- **Semantic (similarity) matching** — threshold knob + false positives buy
  nothing recall needs.
- **Cross-session recall** — requires a new query store and replays queries
  against a blank state; the Langfuse trace store already persists queries
  for audit purposes.

## Consequences

- Zero invalidation surface: data refreshes and index rebuilds cannot stale
  anything.
- Zero latency win: repeated queries pay full pipeline price by design.
- Any future caching proposal must first instrument per-stage latency and
  demonstrate recall usage via the `recalled` telemetry added here.
