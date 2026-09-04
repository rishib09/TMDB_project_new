# AGENTS.md — Maya (TMDB RAG & Evaluation Harness)

Vocabulary: use the ubiquitous language in CONTEXT.md. ADRs 0004–0006 are in force.

## Ticket ritual (every implementation ticket)

Before writing any code, present and get approval:

1. **Overview** of the ticket — what and why, in plain language.
2. **Probable solutions** — brief options with a recommendation.
3. **File/function manifest** — everything that will be created or updated.
4. **Signatures** — for every NEW class/function, show its signature.

Then wait for explicit approval. Model-proposes-code-disposes (ADR 0005) governs design.

## Testing protocol (three tiers, this order)

1. **Adversarial tests first** when fixing a defect — write the failing test against
   current code, then fix.
2. **Unit/mock tests** for pure logic and seams (pure domain layer per ADR 0006).
3. **Live integration tests** last — real LLM/router runs under dotenvx, with
   `PYTHONIOENCODING=utf-8`. Live suites must pass before any close comment.

Every defect fix ships with a test that fails on the old code.

## Libraries and code of conduct

- `requirements.txt` is frozen. **Never add new libraries** without an approved,
  explicit decision. Never reinvent what an existing library or the standard
  library already provides.
- Keep the codebase **lean and human-readable**. Small modules, pure functions
  where possible, one responsibility each — modularity is what makes the
  three-tier testing possible.
- Tunables (thresholds, model ids, cache knobs) go in Experiment Config, never
  hardcoded (ADR 0004).
- Telemetry: if it matters, it appears in the Trace. Fail-open must be explicit
  and recorded.
- Run commands: `./.venv/Scripts/python.exe -m pytest` offline;
  `npx @dotenvx/dotenvx run --` for anything needing env. Windows bash.
- Scratch scripts live in `.scratch/` and are deleted before commit.

## Standing rule

**Before creating anything — verify it doesn't already exist, then take
explicit permission.** This applies to files, features, helpers, and stores.
