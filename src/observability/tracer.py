"""Dual-mode observability for Maya (issue #5): Langfuse cloud + local memory.

Library-first: cloud tracing is the stock ``langfuse.langchain.CallbackHandler``
passed into LangGraph's ``config={'callbacks': [...]}``. The local mode is a
bounded in-memory ring so tests, the trace inspector (#5 UI), and offline
runs always have traces even when Langfuse keys are absent. Cloud
degradation never breaks Maya — missing keys or missing optional deps
degrade to local-only (same graceful pattern as the flashrank reranker).

Note: langfuse's LangChain integration (v4) requires the full ``langchain``
package (requirements.txt); without it the manager stays local-only.
"""

import os
import uuid
from collections import deque
from datetime import UTC, datetime

#: Bounded local ring — prevents unbounded growth in long sessions.
MAX_LOCAL_TRACES = 100


class DualModeObservabilityManager:
    """Manages cloud (Langfuse) and local (in-memory) trace capture."""

    def __init__(self, session_id: str | None = None, max_local_traces: int = MAX_LOCAL_TRACES):
        self.session_id = session_id or "local-session"
        self._local_traces: deque[dict] = deque(maxlen=max_local_traces)
        self._handler: object | None = self._init_cloud_handler()

    def _init_cloud_handler(self, trace_id: str | None = None) -> object | None:
        """Stock Langfuse handler; None when keys or deps are missing (local-only mode).

        When ``trace_id`` is given (issue #9), the handler attaches the whole
        LangChain run to OUR per-turn trace so feedback scores can target the
        exact turn — the id is minted by ``new_turn_trace`` via the Langfuse
        client, keeping 1:1 trace/score correlation.
        """
        if not (
            os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
        ):
            return None
        try:
            from langfuse.langchain import CallbackHandler
            from langfuse.types import TraceContext

            kwargs: dict = {
                "session_id": self.session_id,
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            }
            if trace_id:
                kwargs["trace_context"] = TraceContext(trace_id=trace_id)
            return CallbackHandler(**kwargs)
        except Exception:
            return None

    @property
    def cloud_enabled(self) -> bool:
        """True when a Langfuse handler is active."""
        return self._handler is not None

    def callbacks(self) -> list:
        """LangGraph runnable config callbacks — empty list in local-only mode."""
        return [self._handler] if self._handler else []

    def new_turn_trace(self) -> str:
        """Mints the per-turn trace id and rebinds the cloud handler to it (#9).

        Cloud mode: the id comes from the Langfuse client so scores can attach
        to the same trace. Local mode: a uuid4 — same contract, SQLite-only.
        """
        trace_id = ""
        if self.cloud_enabled:
            try:
                import langfuse

                trace_id = langfuse.get_client().create_trace_id()
            except Exception:
                trace_id = ""
        if not trace_id:
            trace_id = uuid.uuid4().hex
        self._trace_id = trace_id
        self._handler = self._init_cloud_handler(trace_id=trace_id) or None
        return trace_id

    @property
    def current_trace_id(self) -> str:
        """Trace id of the current (latest minted) turn."""
        return getattr(self, "_trace_id", "")

    def record_local(self, node: str, payload: dict) -> None:
        """Captures one node execution in the bounded local ring."""
        self._local_traces.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": self.session_id,
                "node": node,
                "payload": payload,
            }
        )

    def traces(self) -> list[dict]:
        """Readout of local traces (oldest first) for tests and the trace UI."""
        return list(self._local_traces)

    def flush(self) -> None:
        """Flushes cloud traces; no-op in local-only mode."""
        if self._handler:
            self._handler.langfuse.flush()
