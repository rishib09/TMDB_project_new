# 11. Synthesis Voice: Warmth in Content, Neutrality in Mechanics

Status: **Accepted** (2026-09-28, user-approved via issue #117).

We are recording Maya's synthesis voice as a product contract: **warmth in
content, neutrality in mechanics.** Maya is a charming movie guide about
*movies*; she is a precise instrument about *herself*. The contract ships
in the synthesis persona (`src/maya/prompts.py::MECHANICS_RULE`) — the
canonical imperative rendering; the rules below summarise it — so it is
present in every system prompt, and #114 implements its enforcement
(adversarial tests + the `ranking_basis` trace field).

### Why this decision was made:

- **Three visitor-facing incidents in one evening** (feedback inbox #75,
  comments 5875430734 / 5875796867): an excluded title presented as a
  result ("disqualified on a technicality" — the visitor read it as a
  returned movie), invented padding ("classics wing" — content beyond the
  retrieved set), and jokes about system internals ("junk drawer",
  "escorted out") that made a *successful* filter-clear look broken.
- **The same free-form voice measurably hurts evaluation:** 11/200 judge
  fail-opens in the R1 v1 baseline (`production_…023206Z` disclosures, #107)
  trace to verbosity and invented content.
- The voice was a product decision smuggled inside a prompt. Per
  model-proposes-code-disposes (ADR 0005), product decisions are written
  down; #114's enforcement needs an authority to cite.

### The decision — the contract (shipping as `MECHANICS_RULE`):

1. **Personality lives in content**: per-movie hooks, conversational flow,
   follow-ups, apologies on empty results.
2. **Mechanics are stated plainly or not at all**: state transitions,
   active filters, exclusions, ranking basis are never joked about, never
   embellished, never dramatized.
3. **Grounding contract (enforced, not aspirational)**: only retrieved
   titles may be named; excluded/filtered titles are never presented as
   results; ranking basis is disclosed on ranked turns; no content beyond
   the retrieved set — no invented variety.

### Alternatives considered:

- **Status quo (free-form witty voice)** — rejected: measured user
  confusion plus 11 judge fail-opens in one baseline run.
- **Skeleton + model-blurbs** (code writes all connective tissue, the model
  writes only per-movie one-liners) — strongest grounding and cheaper, but
  loses conversational flow; **kept as the named fallback** if this
  contract proves unenforceable in practice.
- **No synthesis (UI renders structured results)** — rejected: most
  reliable, worst conversation product.

### Consequences:

- **#114** is the enforcement ticket: the no-excluded-titles rule and the
  ranking-basis disclosure join the prompt's hard rules; a
  `voice_contract_violations` checker guards the contract offline; the
  `synthesize` trace records `ranking_basis` so Evals can assert disclosure.
- **#115** (acknowledgments) and **#116** (era enforcement) are downstream
  beneficiaries: plain mechanics statements remove the ambiguity that
  produced both reports.
- **Synthesis-prompt changes are experiment-identity changes** (ADR 0010 /
  #107 protocol): any edit to the persona or rules requires disclosure and
  re-measurement before R2 comparison claims are made on top of it.
- Excluded-title mentions on retrieval turns stay **trace-only** (the CWA
  precedent: log, never rewrite grounded prose); the no-retrieval discard
  gate remains the only response-replacing enforcement.
