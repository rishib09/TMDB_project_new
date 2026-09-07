# ADR 0008: Embedding provider seam + column presets (decoupled retrieval axes)

**Status:** Accepted — matrix verdict recorded 2026-09-04
**Date:** 2026-09-04
**Ticket:** #11 Phase 1 (spec in issue thread; Lab visualization deferred to #30)

## Context

Since #14, dense retrieval "tiers" weld three decisions into one name: the column set, the token budget, and the embedding model (all local fastembed/ONNX). Indexing the largest tier locally takes impractically long, the user cannot see what the embedder reads, and trying a better model requires a code change. The user's own suspicion — that most serialized text is noise for retrieval — was unmeasurable.

## Decision

1. **One new seam — `EmbeddingProvider`** (indexing layer): `name`, `dimensions`, `max_tokens`, `embed(texts)`, `token_counter()`. Two backends:
   - `FastembedProvider` — local ONNX, offline default (tests, Hugging Face Spaces, free)
   - `OpenRouterEmbeddingProvider` — cloud via OpenRouter's OpenAI-compatible `POST /embeddings` using the existing `OPENROUTER_API_KEY` and the already-pinned `openai` client. **Zero new keys, zero new libraries** (verified live: 33 models, including free tiers).
2. **Column presets decouple the axes** (domain layer, pure):
   - `minimal`: overview + keywords + genres + title+year + director — **no cast, no tagline**: the pure vibe/theme document
   - `full`: minimal + all top-10 cast (TMDB billing `order`, verified contiguous) + tagline when present
   - Financials, ratings, runtime never enter text (SQL material; Router's MetadataFilterCriteria path owns them)
3. **Budget derives from the chosen model's window** (`provider.max_tokens`), not a tier name. #14's tokenizer-exact packing invariant is preserved; cloud providers without exposed tokenizers pack via a conservative char-estimate counter.
4. **Collection naming encodes both axes**: `<preset>_<model-profile>` — every benchmarked cell stays searchable and reproducible.
5. **Fail-closed providers** (the Langfuse lesson): a provider error propagates; there is NO silent fallback to a different model's vectors. Missing key = loud construction error. Model↔collection pairing is enforced on search (cross-space queries refused).
6. **Legacy tier path preserved bit-for-bit**: `to_dense_text(tier=...)` and the fastembed tier profile path are untouched; already-built collections (v1_0/v1_1/v1_2) remain valid. Legacy tiers are NOT preset names (passing one as a preset is an error).

## Measurement — THE VERDICT (complete matrix, 2026-09-04)

14 golden plot-description queries · 9,119 Movie Records · presets minimal/full ·
all cloud cells via OpenRouter (user direction: no local model runs):

| Cell | Hit@5 | MRR@5 | Build | Cost/pair |
|---|---|---|---|---|
| **gemini_embedding_2 · minimal** | **100%** | 0.952 | 609s | $0.06 |
| **gemini_embedding_2 · full** | **100%** | **0.964** | 608s | $0.08 |
| voyage_4_lite · minimal | 79% | 0.714 | 276s | $0.006 |
| voyage_4_lite · full | 79% | 0.696 | 306s | $0.008 |
| nemotron_free · minimal | 71% | 0.560 | 492s | free |
| nemotron_free · full | 71% | 0.631 | 522s | free |
| bge_m3 · minimal | 71% | 0.613 | 538s | $0.003 |
| lfm_free · minimal | 43% | 0.429 | 347s | free |
| lfm_free · full | 43% | 0.262 | 418s | free |
| bge_m3 · full | **50%** | 0.417 | 692s | $0.004 |

### Findings

1. **Production default: `full_gemini_embedding_2`** (user decision) — the only
   model to find every golden movie, best MRR on the person-rich preset. Wired
   into `MayaSession` as the default `rag_version` + search provider.
2. **The two-lever lesson (the teaching result):** preset winner FLIPS with
   model strength. Weak embedder (lfm 350M): minimal wins big (0.429 vs 0.262)
   — cast/tagline dilute a small model's vector. Strong embedders (nemotron
   1B: 0.631 vs 0.560; gemini: 0.964 vs 0.952): full wins — the person signal
   is USED, not noise. Columns and model must be chosen together.
3. **bge_m3's full-cell collapse** (71% → 50%) is the strongest single proof:
   same columns, same model family — the person names actively hurt it.
4. **bge_m3 at 20x-cheaper than gemini performed identically to the FREE
   nemotron** (71%) — price bought nothing; dropped from the registry.
5. voyage_4_lite: preset-immune (79% both) but never competitive with gemini.

### Final registry (user-approved)

- `lfm_free` — the floor: demonstrates the small-model failure mode (kept for
  the showcase; free)
- `nemotron_free` — the free workhorse: 71% at zero cost (kept; free)
- `gemini_embedding_2` — **the ceiling and PRODUCTION DEFAULT** (kept; ~$0.08
  per full rebuild — ~$4/year at weekly refreshes)
- bge_m3, voyage_4_lite — dropped: no quality niche that free models don't fill
- Legacy local profiles remain for the app's pre-#11 collections only.

### Measurement notes

- Windows verified against the live catalog: lfm 512, gemini 8192, bge_m3 8194,
  voyage family 32000, nemotron 32768. Only lfm's window ever bound; the
  adaptive path self-corrected from the server's own 400 token report.
- Live contract fixes forced by real runs (all fail-closed, zero silent
  degradation): Nvidia rejects base64 encoding_format (pinned float); Google
  caps batches at 100 (batch size lowered); lfm window self-healed via the
  400 report; 1 truncation event counted and reported, never silent.
- Alternative-considered section (template variants, query-side prefixes,
  Google/OpenAI direct keys) retained below.

## Alternatives considered

- **Template variants as a third axis** (end-echo, bare labels, synopsis-first): deferred — multiplies the matrix; revisit after columns × models exists.
- **Query-side instruction prefixes**: touches retrieval runtime, not indexing; deferred.
- **Hardcoding cloud calls inline in the store**: rejected — untestable offline, violates the seams that make three-tier testing possible.
- **Google AI Studio / OpenAI direct keys**: unnecessary once OpenRouter's embeddings endpoint was verified — one existing key covers all cloud cells.

## Consequences

- Offline tests drive the store through a deterministic fake provider — no network, no ONNX, no API spend (FakeProvider pattern mirrors FakeRouter/FakeEngine).
- Cloud cells cost effectively cents per build (9,119 documents); free-tier cells cost nothing.
- The `Any`-typed fastembed tokenizer dependency is owned by the provider seam; the domain layer stays Pydantic-pure (ADR 0006).
- Experiment Config can select any cell via version name + provider factory (ADR 0004).
