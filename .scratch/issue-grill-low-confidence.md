# Grill: Low-Confidence Router Policy — Retry vs. Config Knob vs. UI Transparency

## Status

`grilling` — open design questions to stress-test before implementation. **Do not implement until the decision gates below are satisfied.**

## Background (what exists today, Issue #3)

`MayaRouter.route()` currently degrades via `_heuristic_fallback()` on two triggers:

1. **API unreachable / schema failure** — uncontroversial, keep.
2. **LLM confidence < threshold (default 0.5)** — theoretically questionable: we swap a low-confidence *expert* answer for a dumb-but-predictable regex heuristic. Rationale: low confidence usually means the LLM is confused, and structured output can be *quietly wrong* in that regime. Predictable degradation > unpredictable accuracy.

The fallback decision carries `confidence=0.1` + `reasoning="Heuristic fallback: ..."` so downstream never mistakes it for a solid answer, and session exclusions are still merged.

## The tension to grill

A 30%-confident LLM answer is probably still better than a keyword regex. Is fallback-on-low-confidence actually the right default, or are we trading real accuracy for perceived safety?

---

## Questions to grill (in order)

### Q1 — Can we improve performance just by changing the confidence norm?

- Is `confidence` from a temp=0.0 structured-output call even *calibrated*? Llama-3.2-3b may habitually output 0.6–0.8 (threshold = dead code) or 0.3 (fallback fires constantly).
- What norm/threshold maximizes end-metric performance (Hit Rate@K, MRR@K — not just routing accuracy)?
- Should the threshold be **per-model** (like `model_ceilings` in `ExperimentConfig.validate_token_budget_against_model`) rather than global?
- Gate: needs the **confidence distribution from live runs** across the 7 intents before any number is defensible.

### Q2 — Option A: retry-once-before-fallback?

- Low confidence → re-invoke once with a rephrased/expanded query or a stricter prompt; fall back only if the retry also fails.
- Cost: doubles router latency + token spend exactly in the case where the first call was already shaky.
- Does retrying make sense at all when the router model is a 3B at temp 0.0 — would a *different* model (fallback chain, e.g. retry with synthesis model) be smarter?
- Gate: measure how often live confidence actually lands below threshold. If < 2% of queries, retry machinery is over-engineering.

### Q3 — Option B: `low_confidence_policy` knob in ExperimentConfig?

- Add to the Memory & Guardrails section: `low_confidence_policy: "fallback" | "keep_with_flag" | "retry_once"` + `confidence_threshold: float`.
- Fits ADR #4 exactly: this is an architectural knob, benchmarkable across policies, toggleable from the Sidebar Lab / `/admin`.
- Grill: is this a **benchmark-relevant** knob (deserves a slot in `ExperimentConfig` + eval matrix) or a **deployment setting** (belongs in env/config, not the experimentation plane)?
- Gate: Q1 + Q2 conclusions determine which policies are even worth exposing.

### Q4 — Transparency: how (and whether) to show the low-confidence policy on the UI?

This is the core UX grilling target:

1. **Do we show it to the end user at all?** Options:
   - Never (silent) — user sees a confident answer either way.
   - Subtle signal — e.g. a small badge / footnote: "answered with reduced certainty".
   - Full honesty — Maya *says it in-character*: "I'm not fully sure I got that — here's my best shot."
2. **Does showing it hurt trust more than it helps?** Does "reduced certainty" on a movie recommendation even mean anything to a casual user? Or only to evaluators?
3. **Admin-level only?** The Trace Inspector (Observability tab) already renders span-level detail — confidence, fallback trigger, reasoning should unquestionably appear *there*. Is the Trace Inspector the right and *sufficient* level of transparency, with the chat surface staying clean?
4. **If in-character disclosure wins:** how does it interact with the persona (Issue #10)? "Upbeat, sassy, never overbearing" + uncertainty disclosure could produce charm ("my gut's a little fuzzy on this one...") — or constant hedging that violates the voice spec. Who writes that line, and is it prompt (Issue #10) or programmatic (badge)?
5. **Langfuse symmetry:** should the fallback/low-confidence event also emit a `langfuse.score()` / event so it's visible in cloud telemetry, not just the local inspector?

### Q5 — Success metrics for the decision itself

Whichever policy wins, how do we *know* it won? Proposals:

- % of queries hitting fallback (should be low but nonzero)
- Routing accuracy on the ground-truth suite per policy
- End-to-end retrieval metrics delta (fallback path returns unfiltered SEMANTIC_SEARCH — does it actually hurt retrieval?)
- User feedback (thumbs down rate) correlation with low-confidence turns — measurable once Issue #9 feedback store is live

---

## Decision gates

- [ ] **Gate 1:** Live test run of Issue #3 executed; confidence distribution per intent recorded.
- [ ] **Gate 2:** Threshold/norm analysis done (Q1) — per-model table if needed.
- [ ] **Gate 3:** Retry frequency estimate done (Q2) — is it worth the latency?
- [ ] **Gate 4:** Policy knob scope decided (Q3) — benchmark knob vs deployment setting.
- [ ] **Gate 5:** UI transparency decision (Q4) — user-facing / admin-only / in-character, with persona sign-off from Issue #10.

## Dependencies

- **Blocked by:** #3 (live run data)
- **Relates to:** #8 (ExperimentConfig control plane exposure), #9 (feedback correlation data), #10 (persona interaction if in-character disclosure), #5 (Trace Inspector rendering surface)

## Non-goals

- No changes to the API-failure fallback path (that one is settled).
- No new LLM provider dependencies.
