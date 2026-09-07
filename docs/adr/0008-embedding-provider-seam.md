# ADR 0008: Embedding provider seam + column presets (decoupled retrieval axes)

**Status:** Accepted
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

## Measurement (the tier verdict instrument)

`scripts/benchmark_dense_matrix.py` builds each preset × model cell and measures golden Hit@5/MRR@5 over the #14 golden queries. The matrix includes free OpenRouter tiers first (`lfm-2.5-embedding-350m:free`, `nemotron-3-embed-1b:free`), then cheap paid (`google/gemini-embedding-2`, `baai/bge-m3`, `voyage-4-lite`), plus the three local models. **Whether the legacy middle tier earns its keep is decided by this table, not intuition** — verdict recorded here when the matrix run completes.

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
