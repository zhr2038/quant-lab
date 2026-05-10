import ast
import io
import tokenize
from pathlib import Path

SOURCE_ROOT = Path("src/quant_lab")
FORBIDDEN_CONCEPTS = {
    "place_order",
    "create_order",
    "cancel_order",
    "amend_order",
    "private_key",
    "api_secret",
    "withdraw",
    "transfer_funds",
    "execute_trade",
    "live_order_mutation",
}


def test_no_forbidden_execution_surface_in_python_sources():
    offenders: list[str] = []

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        offenders.extend(_forbidden_definitions(path, tree))
        offenders.extend(_forbidden_route_paths(path, tree))
        offenders.extend(_forbidden_implementation_tokens(path, text))

    assert offenders == []


def _forbidden_definitions(path: Path, tree: ast.AST) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            lowered = node.name.lower()
            for concept in FORBIDDEN_CONCEPTS:
                if concept in lowered:
                    offenders.append(
                        f"{path}:{node.lineno} forbidden definition name {node.name!r}"
                    )
    return offenders


def _forbidden_route_paths(path: Path, tree: ast.AST) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr.lower() not in {"get", "post", "put", "patch", "delete", "options"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        route_path = node.args[0].value
        if not isinstance(route_path, str):
            continue
        lowered = route_path.lower()
        for concept in FORBIDDEN_CONCEPTS:
            if concept in lowered:
                offenders.append(f"{path}:{node.lineno} forbidden route path {route_path!r}")
    return offenders


def _forbidden_implementation_tokens(path: Path, text: str) -> list[str]:
    offenders: list[str] = []
    stream = io.StringIO(text)
    for token in tokenize.generate_tokens(stream.readline):
        if token.type not in {tokenize.NAME}:
            continue
        lowered = token.string.lower()
        for concept in FORBIDDEN_CONCEPTS:
            if concept in lowered:
                offenders.append(
                    f"{path}:{token.start[0]} forbidden implementation identifier {token.string!r}"
                )
    return offenders
