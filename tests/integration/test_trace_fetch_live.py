"""Live fetch-back test (issue #31): emit a real trace, fetch it back.

Requires LANGFUSE_* keys (run under dotenvx). Polls with a deadline because
Langfuse ingestion is asynchronous (documented 15-30s lag).
"""

import os
import time

import pytest

from src.observability.trace_fetch import fetch_trace_tree

pytestmark = pytest.mark.live

INGESTION_DEADLINE_S = 90
POLL_INTERVAL_S = 5

needs_keys = pytest.mark.skipif(
    not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
    reason="LANGFUSE_* keys not configured",
)


@needs_keys
def test_emitted_trace_fetches_back_with_io():
    from langfuse import get_client

    client = get_client()
    trace_id = client.create_trace_id()
    with client.start_as_current_observation(
        name="test-root",
        as_type="span",
        input={"query": "fetch-back live test"},
        trace_context={"trace_id": trace_id},
    ) as span:
        with client.start_as_current_observation(
            name="test-generation",
            as_type="generation",
            model="test-model",
            input=[{"role": "system", "content": "You are Maya."},
                   {"role": "user", "content": "ping"}],
        ) as gen:
            gen.update(output="pong")
        span.update(output={"status": "ok"})
    client.flush()

    deadline = time.time() + INGESTION_DEADLINE_S
    tree = None
    while time.time() < deadline:
        tree = fetch_trace_tree(trace_id)
        if tree is not None and tree.roots:
            flat = _flatten_names(tree.roots)
            if "test-generation" in flat:
                break
        time.sleep(POLL_INTERVAL_S)

    assert tree is not None, "trace never became fetchable within deadline"
    names = _flatten_names(tree.roots)
    assert "test-root" in names and "test-generation" in names
    gen_node = _find(tree.roots, "test-generation")
    assert gen_node.type == "GENERATION"
    assert gen_node.model == "test-model"
    assert any("Maya" in str(m) for m in (gen_node.input or [])), "system prompt missing"
    assert "pong" in str(gen_node.output)


def _flatten_names(nodes) -> list[str]:
    out = []
    for n in nodes:
        out.append(n.name)
        out.extend(_flatten_names(n.children))
    return out


def _find(nodes, name):
    for n in nodes:
        if n.name == name:
            return n
        found = _find(n.children, name)
        if found:
            return found
    return None
