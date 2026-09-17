# AGENTS.md — Maya (TMDB RAG & Evaluation Harness)

Vocabulary: use the ubiquitous language in CONTEXT.md. ADRs 0004–0006 are in force.

## Ticket ritual (every implementation ticket)

Before writing any code, present and get approval:

1. **Overview** of the ticket — what and why, in plain language.
2. **Library check** — a table, one row per capability the ticket needs:
   `capability | installed API (package, class/function, version) | import proof | or: none installed → 2–3 alternatives with pros, cons, package cost`.
   Start from `docs/library-map.md`; verify with an import in `.venv` (pins are
   exact, see `requirements.txt`). An option that reimplements an installed API
   must say why. No approval without this table.
3. **Probable solutions** — brief options with a recommendation, each naming
   the row of the library check it builds on.
4. **File/function manifest** — everything that will be created or updated.
5. **Signatures** — for every NEW class/function, show its signature and the
   library API it wraps or calls.

Then wait for explicit approval. Model-proposes-code-disposes (ADR 0005) governs design.

## Testing protocol (three tiers, this order)

1. **Adversarial tests first** when fixing a defect — write the failing test against
   current code, then fix.
2. **Unit/mock tests** for pure logic and seams (pure domain layer per ADR 0006).
3. **Live integration tests** last — real LLM/router runs under dotenvx, with
   `PYTHONIOENCODING=utf-8`. Live suites must pass before any close comment.

Every defect fix ships with a test that fails on the old code.

## Libraries and code of conduct

- **Installed libraries first.** Before designing any logic, find the installed
  package API that covers it and cite it (the library check in the ticket
  ritual). `docs/library-map.md` lists the covered capabilities; the venv is the
  truth for what is installed, and versions there are the exact pins in
  `requirements.txt`. Reimplement only with a stated reason in the manifest.
- **When nothing installed fits**, do not build from scratch by default: present
  two or three options (a new package, a standard-library route, a hand-written
  route) with pros, cons and package cost, and let the user decide.
- `requirements.txt` is frozen at exact versions. **Never add or bump a library**
  without an approved, explicit decision. Never reinvent what an existing
  library or the standard library already provides.
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

## Git workflow — branch + PR, never direct to main (mandatory)

- **One branch per ticket**, branched from latest `main`: `git checkout -b <issue#>-<short-slug>`.
- **Never commit directly to `main`. Never push to `main`.** All work lands via PR.
- **Open a PR against `main`** referencing the ticket number, with a summary of
  what changed and what to verify, and a line naming any new function that
  reimplements an installed API, with the manifest's reason.
- **Never merge the PR — not automatically, not "while I'm at it".** Wait for
  the user to explicitly approve and merge. The agent's job ends at
  "PR is ready for your review: <link>".
- **`git add` only files you yourself changed — never `git add -A`/`git add .`**
  while other sessions may be active. Concurrent sessions share this working
  tree; sweeping staging areas has already swept foreign work into a foreign
  commit once (2026-09-04, see #12/#13 attribution comments).
- **Pull/merge latest `main` before starting** a ticket — multiple sessions
  work this repo concurrently.
- (Bootstrap exception, 2026-09-04: this very section was committed to `main`
  directly at the user's instruction. Every ticket after it uses the flow above.)

## Standing rule

**Before creating anything — verify it doesn't already exist, then take
explicit permission.** This applies to files, features, helpers, and stores.
