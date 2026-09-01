"""Import-lint guard for ADR 0006: src/domain/ must stay pure Pydantic.

Fails the suite the moment any domain module imports a framework
(langgraph, langchain*, streamlit, chromadb, fastembed, ...), so the
MayaGraphState-in-domain violation from issue #2 can never silently return.
"""

import ast
from pathlib import Path

import pytest

DOMAIN_DIR = Path(__file__).parents[2] / "src" / "domain"

#: Frameworks and infrastructure the domain layer must never touch.
BANNED_ROOTS = (
    "langgraph",
    "langchain",
    "streamlit",
    "chromadb",
    "fastembed",
    "openai",
    "langfuse",
)


def _imported_modules(tree: ast.AST) -> list[str]:
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


@pytest.mark.unit
@pytest.mark.parametrize("py_file", sorted(DOMAIN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_domain_module_imports_no_frameworks(py_file: Path):
    for module in _imported_modules(ast.parse(py_file.read_text(encoding="utf-8"))):
        assert not module.startswith(BANNED_ROOTS), (
            f"{py_file.name} imports '{module}' — ADR 0006 violation: "
            "src/domain/ must stay pure (Pydantic + stdlib only)"
        )
