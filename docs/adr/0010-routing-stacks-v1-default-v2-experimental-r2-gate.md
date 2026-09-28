# 10. Routing Stacks: v1 Remains Default, v2 Sanctioned Experimental, Promotion Gated on R2

We are recording a **two-part decision** on the routing stacks: (1) **now**, v1 (gemini-3.5-flash-lite) remains the production default while v2 (glm-5.3-flash primary, gemini-3.5-flash-lite fallback) is the sanctioned experimental stack — selectable, first-class-tested, and safe to run live; and (2) **promotion of v2 to production default is gated on a repeat measurement (R2)** run after the #120/#121 reference-turn fixes land, compared on the same table as R1.

### Why this decision was made:

- **The R1 measurement (issue #107, all runs validated, 200 turns each) currently favors v1 on every graded axis** — promoting v2 today would mean deciding against our own evidence:

  | run | stack / model | intent | fidelity | no-repeat | api-fallbacks |
  |---|---|---|---|---|---|
  | `production_…023206Z` | v1 / gemini-3.5-flash-lite (fence-fixed) | **0.798** | **0.369** | 1.000 | 3/200 |
  | `custom_c0e9dc24` | v2 / glm-5.3-flash | 0.710 | 0.226 | 1.000 | 0/200 |
  | `custom_29229b16` | v2 / gemini-3.5-flash-lite | 0.710 | 0.177 | 1.000 | 0/200 |
  | `custom_7e91360d` | v2 / gemini-3.8-flash | 0.685 | 0.201 | 1.000 | 0/200 |
  | `custom_1767515a` | v2 / gemma-4-31b | 0.650 | 0.147 | 1.000 | 0/200 |

  Constraint fidelity (the Q13 rule) measures the preference *state* — v2's entire reason to exist — and v1 still leads there (0.369 vs 0.226). An ADR that promoted v2 on this table would be a hope, not a decision.

- **But the evidence has known limits, recorded here as disclosures:** each stack ran once (**n=1** — run-to-run variance unquantified); the golden set (23 conversations, `data/eval_conversations.json`) was authored around v1's funnel and may weight v1's classification strengths; the judge (llama-3.3-70b, 1024 cap per #74) failed open on 11/200 v1 turns, thinning that fidelity column (intent and no-repeat are judge-independent); and the v1 cost figure is unreliable (negative `total_cost_usd` accounting anomaly, #123-adjacent).

- **Live UI evidence since R1 cuts both ways.** The 9-conversation AppTest replay (#118) showed v2/glm holding a 79-turn live session with zero mislabels and the C06 ten-turn exclusion chain intact — but it also surfaced gaps that are **stack-wide, not v2-specific**: ordering requests have no channel to an ORDER BY (#120 — v1's `QueryRoutingDecision` cannot express one either), and reference turns can drop standing attribute filters (#121, with v1-era precedent in #12). Neither stack is "finished"; v2's wounds are ticketed and architecturally closer to fixable (the disposer owns state invariants in code).

- **Keeping both stacks is now cheap; removal is expensive.** The stack selector (`routing_stack` config + `MAYA_ROUTING_STACK` env), per-stack routers, the #118 UI test tier, and 758 green tests make dual-stack maintenance a small, tested surface. Deleting either stack before the gate would destroy the measurement's control group.

- **The fence incident validated the architecture, not the leaderboard.** v1's 48% fleet failure was a transport-format drift repaired in one place (`_FenceTolerantChain`, #111); v2 absorbed the same drift class all along because it owns its parse (#106) and now carries an automatic model fallback (#118). The stacks fail differently under provider drift — that resilience asymmetry belongs in the promotion calculus, which is why the gate re-measures rather than re-litigating R1.

### The decision:

1. **Production default: unchanged.** v1 / gemini-3.5-flash-lite remains the default stack for UI sessions and conversation-mode evals.
2. **v2 is sanctioned experimental:** selectable via `MAYA_ROUTING_STACK=v2` / Experiment Config (`routing_stack`, `v2_router_model` default `glm-5.3-flash`, `v2_router_fallback_model` default `gemini-3.5-flash-lite`), covered by unit, adversarial, AppTest UI tiers, and live-replay acceptance.
3. **The promotion gate (R2):** after #120 (ordering channel) and #121 (reference-turn filter persistence) land, run **n=3 repeats per stack** (v1 and v2/glm) on the golden set — ~$2.50 total at current prices — and compare mean ± spread against R1 on the same axes (intent, path, fidelity, no-repeat, api-fallbacks, cost). **v2 promotes to production default iff** its intent matches or exceeds v1's within the observed spread and its fidelity is not significantly worse, with zero transport failures. R2 results and the promotion (or refusal) land as an amendment to this ADR.
4. **Either way, the losing stack is not deleted.** It remains the control: routing claims in this repo are backed by comparative measurement (map #81), and a single-stack world can no longer produce one.

### Consequences:

- Issue #90 closes with this ADR; the promotion question moves to the R2 amendment (new ticket at gate time).
- #120 and #121 become gate prerequisites — they change the measured behavior and must land before R2, not after.
- The cost-accounting anomaly (#123-adjacent) must be fixed before R2 so the cost axis is comparable.
- CONTEXT.md v2 terms (C13) unblock once the #39 concurrent session's edits land — the ubiquitous language now has an accepted ADR to point at.
- Voice/persona policy (synthesis jokes, #114/#117) is orthogonal to stack promotion and does not block the gate.
- Any future prompt, cap, or parse-ownership change to either stack is an experiment-identity change: disclose in the run envelope and re-measure (per #107/#111 precedent).
