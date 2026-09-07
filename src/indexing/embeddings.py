"""Embedding provider seam (#11 Phase 1): one interface, two backends.

The ONE new seam of ticket #11: local fastembed (ONNX, offline default) and
cloud OpenRouter embeddings (existing key, already-pinned openai client)
both satisfy :class:`EmbeddingProvider`. Everything downstream —
``MovieVectorStore.index_movies`` / ``search`` — accepts a provider, so all
offline tests drive the store with a deterministic fake (no network, no
ONNX).

Fail-closed rule (the Langfuse lesson): a provider error propagates; there
is NO silent fallback to a different model's vectors.
"""

from __future__ import annotations

import math
import os
from typing import Any, Protocol, runtime_checkable

from src.domain.movie import TokenCounter


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural interface: any duck-typed provider works, no inheritance."""

    name: str          # model id as recorded in collection metadata
    dimensions: int    # output vector width (informational; chroma infers)
    max_tokens: int    # packing window — documents never exceed this

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; order-preserving."""
        ...

    def token_counter(self) -> TokenCounter | None:
        """Real tokenizer when available (exact packing); None → char estimate."""
        ...


class CharEstimateCounter:
    """Chars-per-token estimator for providers without an exposed tokenizer.

    Cloud providers don't hand out tokenizers; packing against their window
    uses this estimate. CAUTION (measured 2026-09-04, #11): free lfm's real
    tokenizer packs ~1.7x denser than 4 chars/token on label-dense docs —
    a 400 'exceeds model maximum' mid-build is the signature of trusting
    this estimate as exact. Providers using it MUST pack with a safety
    margin (see OpenRouterEmbeddingProvider.packing_budget).
    """

    #: Measured worst-case density ratio (real tokens ÷ estimated) across the
    #: 9,119-record corpus; 2.0x margin covers label-dense outlier documents.
    SAFETY_FACTOR: float = 2.0

    def __init__(self, chars_per_token: float = 4.0, max_tokens: int = 0):
        self.chars_per_token = chars_per_token
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[: int(max_tokens * self.chars_per_token)]


# --- local backend -----------------------------------------------------------

#: Measured real fastembed windows (issue #14, 2026-08-31): model-card
#: numbers overstate; these are the tokenizer-measured truths.
FASTEMBED_MEASURED_WINDOWS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 128,
    "snowflake/snowflake-arctic-embed-s": 512,
    "jinaai/jina-embeddings-v2-base-en": 1024,
}

#: Module-level ONNX cache shared by every FastembedProvider instance —
#: loading the same model twice would double RAM for zero benefit.
_FASTEMBED_CACHE: dict[str, Any] = {}
_COUNTER_CACHE: dict[str, Any] = {}


class FastembedProvider:
    """Local ONNX embeddings (default; offline, free, no API dependency)."""

    def __init__(self, model_name: str, max_tokens: int | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        self.max_tokens = max_tokens or FASTEMBED_MEASURED_WINDOWS.get(model_name, 512)
        if model_name not in _FASTEMBED_CACHE:
            _FASTEMBED_CACHE[model_name] = TextEmbedding(model_name=model_name)
        self._model = _FASTEMBED_CACHE[model_name]
        if model_name not in _COUNTER_CACHE:
            _COUNTER_CACHE[model_name] = FastembedTokenCounter(
                self._model.model.tokenizer
            )
        self._counter: FastembedTokenCounter = _COUNTER_CACHE[model_name]
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(next(self._model.embed(["dimension probe"])))
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [emb.tolist() for emb in self._model.embed(texts)]

    def token_counter(self) -> TokenCounter:
        return self._counter


class FastembedTokenCounter:
    """TokenCounter backed by the target embedding model's real tokenizer.

    Truncation cuts on token boundaries using encoder offsets, preserving the
    original text (casing/punctuation) instead of round-tripping through
    decode — no silent truncation, no character estimates (issue #14).

    Padding matters: some fastembed tokenizers (e.g. MiniLM) pad every
    encoding to a fixed sequence length, so raw ``len(ids)`` is a constant,
    not a content length. Always count via the attention mask.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        encoded = self._tokenizer.encode(text)
        return sum(encoded.attention_mask)

    def truncate(self, text: str, max_tokens: int) -> str:
        encoded = self._tokenizer.encode(text)
        real_length = sum(encoded.attention_mask)
        if real_length <= max_tokens:
            return text
        cut = encoded.offsets[max_tokens - 1][1]
        prefix = text[:cut]
        # Trim to a clean word boundary.
        return prefix.rsplit(" ", 1)[0] if " " in prefix else prefix


# --- cloud backend -----------------------------------------------------------

#: Model registry for the #11 benchmark matrix. ``max_tokens`` is the packing
#: window granted to the preset serializer (conservative where unverified —
#: the adaptive cloud path self-corrects from the server's own 400 report).
#: LOCAL cells are excluded from benchmark defaults per user direction
#: (2026-09-04): the matrix is cloud-only; fastembed stays the offline
#: default for the app's existing collections, not for matrix cells.
MODEL_PROFILES: dict[str, dict[str, str | int]] = {
    # local (offline default for the APP — not benchmark cells)
    "minilm_local": {"backend": "fastembed", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    "bge_small_local": {"backend": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
    "jina_v2_local": {"backend": "fastembed", "model": "jinaai/jina-embeddings-v2-base-en"},
    # cloud via OpenRouter (verified live 2026-09-04; free tiers first)
    "lfm_free": {"backend": "openrouter", "model": "liquid/lfm-2.5-embedding-350m:free", "max_tokens": 512},
    "nemotron_free": {"backend": "openrouter", "model": "nvidia/nemotron-3-embed-1b:free", "max_tokens": 2048},
    "gemini_embedding_2": {"backend": "openrouter", "model": "google/gemini-embedding-2", "max_tokens": 2048},
    "bge_m3": {"backend": "openrouter", "model": "baai/bge-m3", "max_tokens": 2048},
    "voyage_4_lite": {"backend": "openrouter", "model": "voyageai/voyage-4-lite", "max_tokens": 2048},
}

#: Benchmark default = cloud-only (user direction: no local model runs).
BENCHMARK_PROFILES: list[str] = [
    "lfm_free", "nemotron_free", "gemini_embedding_2", "bge_m3", "voyage_4_lite",
]


def provider_from_profile(profile_name: str) -> EmbeddingProvider:
    """Builds a provider from MODEL_PROFILES — the benchmark matrix's factory."""
    profile = MODEL_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(
            f"Unknown model profile '{profile_name}'. Valid: {sorted(MODEL_PROFILES)}"
        )
    if profile["backend"] == "fastembed":
        return FastembedProvider(model_name=str(profile["model"]))
    return OpenRouterEmbeddingProvider(
        model=str(profile["model"]),
        max_tokens=int(profile.get("max_tokens", 2048)),
    )


class OpenRouterEmbeddingProvider:
    """Cloud embeddings via OpenRouter's OpenAI-compatible endpoint.

    Uses the already-pinned ``openai`` client with the OpenRouter base URL
    and the existing OPENROUTER_API_KEY — zero new keys, zero new libraries.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 4,
    ):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set — refusing to construct a cloud "
                "embedding provider that cannot work (fail-closed)."
            )
        self._client = OpenAI(
            base_url=base_url or self.DEFAULT_BASE_URL, api_key=key
        )
        self.name = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.dimensions = 0  # learned from the first embedding response
        # Truncation telemetry (#11): a build must be able to SAY whether any
        # document was cut, and by how much — silent truncation is forbidden.
        self.truncation_events = 0
        self.max_truncated_tokens = 0
        self.real_tokens_seen = 0  # from response.usage, when the API reports it

    def packing_budget(self) -> int:
        """Budget granted to the packer: the model's REAL window, no halving.

        The window itself is self-healing: the first 400 window-error reports
        the model's true maximum and ``max_tokens`` is corrected in place.
        Documents that genuinely exceed the window are truncated minimally,
        adaptively, and LOUDLY (truncation_events) — never silently, never
        preemptively halved (user direction, #11: follow the model's own
        context token and measure truncation rather than guess it).
        """
        return self.max_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds with rate-limit retry + adaptive window handling.

        Strategy (user direction, #11): always attempt the FULL document —
        never preemptively shrink. On a window error (400 with the server's
        token report), fall back to per-document embedding for that batch:
        the error's real token count calibrates the true density, the cut is
        minimal (keep ~98% of the window), and every cut increments
        ``truncation_events`` — measurable, never silent.
        """
        try:
            return self._with_rate_retry(lambda: self._embed_batch(texts))
        except Exception as exc:
            if self._parse_window_error(exc) is None:
                raise
            print(f"  [window] batch exceeds the model window ({self.max_tokens} tok) "
                  f"— per-document adaptive embedding for this batch", flush=True)
            return [self._embed_one_adaptive(text) for text in texts]

    def _embed_one_adaptive(self, text: str) -> list[float]:
        """Embeds one document, truncating minimally using the server's own
        token-count report. Never silent: every cut is counted and bounded."""
        for _ in range(3):
            try:
                return self._with_rate_retry(lambda: self._embed_batch([text]))[0]
            except Exception as exc:
                parsed = self._parse_window_error(exc)
                if parsed is None:
                    raise
                real_tokens, model_max = parsed
                # The server's numbers are ground truth: correct the window,
                # measure this document's true density, cut just enough.
                self.max_tokens = model_max
                density = real_tokens / max(1, len(text))
                keep_chars = int(len(text) * (model_max * 0.98) / real_tokens)
                self.truncation_events += 1
                self.max_truncated_tokens = max(self.max_truncated_tokens, real_tokens)
                print(f"  [truncate] doc {real_tokens} tok > {model_max} window "
                      f"(density {density:.2f}x) — cut to {model_max * 0.98:.0f} tok",
                      flush=True)
                text = text[:keep_chars]
        raise RuntimeError(
            f"document could not be packed inside the {self.max_tokens}-token "
            f"window after adaptive truncation"
        )

    @staticmethod
    def _parse_window_error(exc: Exception) -> tuple[int, int] | None:
        """Extracts (real_tokens, model_max) from a 400 window error, if present."""
        import re

        status = getattr(exc, "status_code", None)
        if status is not None and status != 400:
            return None
        match = re.search(r"has (\d+) tokens, exceeding the model maximum of (\d+)",
                          str(exc))
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.name, input=texts)
        usage = getattr(response, "usage", None)
        if usage is not None and getattr(usage, "prompt_tokens", None):
            self.real_tokens_seen += usage.prompt_tokens
        vectors = [item.embedding for item in response.data]
        if vectors and self.dimensions == 0:
            self.dimensions = len(vectors[0])
        return vectors

    def _with_rate_retry(self, attempt_fn):
        """Runs one embedding attempt with rate-limit-aware retry (429 honors
        the documented remedy — never silent degradation; after max_retries
        backoff windows the error propagates, fail-closed)."""
        import time

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return attempt_fn()
            except Exception as exc:  # noqa: BLE001 — narrowed by status check
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or status is None  # transport errors too
                last_error = exc
                if not retryable or attempt == self.max_retries:
                    raise
                wait_s = self._backoff_s(attempt)
                print(f"  [rate-limited] retry {attempt + 1}/{self.max_retries} "
                      f"in {wait_s:.0f}s", flush=True)
                time.sleep(wait_s)
        raise last_error  # unreachable; satisfies type checkers

    def _backoff_s(self, attempt: int) -> float:
        """Backoff spanning whole per-minute windows: 20s, 45s, 90s, 180s."""
        return (20.0, 45.0, 90.0, 180.0)[min(attempt, 3)]

    def token_counter(self) -> TokenCounter | None:
        return CharEstimateCounter(max_tokens=self.max_tokens)
