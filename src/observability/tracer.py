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

    def _init_cloud_handler(self) -> object | None:
        """Stock Langfuse handler; None when keys or deps are missing (local-only mode)."""
        if not (
            os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
        ):
            return None
        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler(
                session_id=self.session_id,
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        except Exception:
            return None

    @property
    def cloud_enabled(self) -> bool:
        """True when a Langfuse handler is active."""
        return self._handler is not None

    def callbacks(self) -> list:
        """LangGraph runnable config callbacks — empty list in local-only mode."""
        return [self._handler] if self._handler else []

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
