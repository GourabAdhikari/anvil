"""Find similar functions and classes across source repositories."""

from __future__ import annotations

import ctypes
import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
_SOURCE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx"}
_IGNORED_DIRECTORIES = {".git", "node_modules", "__pycache__", ".venv", "venv"}
_NODE_TYPES = {
    "python": {"function_definition", "async_function_definition", "class_definition"},
    "javascript": {
        "function_declaration", "generator_function_declaration", "function",
        "arrow_function", "class_declaration", "method_definition",
    },
    "typescript": {
        "function_declaration", "generator_function_declaration", "function",
        "arrow_function", "class_declaration", "method_definition", "abstract_class_declaration",
    },
    "tsx": {
        "function_declaration", "generator_function_declaration", "function",
        "arrow_function", "class_declaration", "method_definition", "abstract_class_declaration",
    },
}


@dataclass(frozen=True)
class _Chunk:
    repo_path: str
    file_path: str
    function_name: str
    source: str


def _language_for_path(path: Path) -> str:
    if path.suffix == ".py":
        return "python"
    if path.suffix == ".ts":
        return "typescript"
    if path.suffix == ".tsx":
        return "tsx"
    return "javascript"


def _parser_from_legacy_bundle(language: str, parser_type: Any, language_type: Any) -> Any:
    """Load grammars from tree-sitter-languages with modern bindings.

    tree-sitter-languages 1.10.x calls the pre-0.25 two-argument Language
    constructor. When paired with tree-sitter 0.26, its public helpers fail;
    the bundled grammar library can still be loaded through its C API.
    """
    import tree_sitter_languages

    bundle = Path(tree_sitter_languages.__file__).with_name("languages.so")
    if not bundle.exists():
        raise RuntimeError("tree-sitter language bundle was not found")

    symbol_name = "tree_sitter_typescript" if language == "tsx" else f"tree_sitter_{language}"
    grammar_function = getattr(ctypes.CDLL(str(bundle)), symbol_name)
    grammar_function.restype = ctypes.c_void_p
    language_object = language_type(grammar_function())

    parser = parser_type()
    if hasattr(type(parser), "language"):
        parser.language = language_object
    else:
        parser.set_language(language_object)
    return parser


def _parser(language: str) -> Any:
    try:
        from tree_sitter_languages import get_parser

        try:
            return get_parser(language)
        except TypeError:
            # Compatibility with tree-sitter >= 0.25 and the older bundled
            # tree-sitter-languages helper.
            from tree_sitter import Language, Parser

            return _parser_from_legacy_bundle(language, Parser, Language)
    except ImportError:
        pass

    try:
        from tree_sitter import Language, Parser
    except ImportError as exc:
        raise RuntimeError("tree-sitter is not installed") from exc

    package_name = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "tsx": "tree_sitter_typescript",
    }[language]
    try:
        grammar = importlib.import_module(package_name)
        if language == "typescript":
            language_object = grammar.language_typescript()
        elif language == "tsx":
            language_object = grammar.language_tsx()
        else:
            language_object = grammar.language()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"A tree-sitter grammar for {language} is not installed"
        ) from exc

    try:
        return Parser(language_object)
    except TypeError:
        parser = Parser()
        if hasattr(type(parser), "language"):
            parser.language = language_object
        else:
            parser.set_language(language_object)
        return parser


def _node_name(node: Any, source: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        for child in node.named_children:
            if child.type in {"identifier", "property_identifier", "type_identifier"}:
                name_node = child
                break
    if name_node is None:
        return "<anonymous>"
    return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")


def _walk(node: Any, source: bytes, language: str, repo_path: str, file_path: str) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    if node.type in _NODE_TYPES[language]:
        chunks.append(
            _Chunk(
                repo_path=repo_path,
                file_path=file_path,
                function_name=_node_name(node, source),
                source=source[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
            )
        )
    for child in node.named_children:
        chunks.extend(_walk(child, source, language, repo_path, file_path))
    return chunks


def _source_files(repo: Path) -> list[Path]:
    if repo.is_file():
        return [repo] if repo.suffix in _SOURCE_EXTENSIONS else []
    files: list[Path] = []
    for path in repo.rglob("*"):
        if path.is_file() and path.suffix in _SOURCE_EXTENSIONS:
            if not any(part in _IGNORED_DIRECTORIES for part in path.relative_to(repo).parts):
                files.append(path)
    return files


def _chunks(repo: Path, repo_label: str) -> list[_Chunk]:
    result: list[_Chunk] = []
    for path in _source_files(repo):
        source = path.read_bytes()
        parser = _parser(_language_for_path(path))
        tree = parser.parse(source)
        result.extend(_walk(tree.root_node, source, _language_for_path(path), repo_label, str(path)))
    return result


@lru_cache(maxsize=1)
def _embedder() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed") from exc
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return SentenceTransformer(MODEL_NAME, device="cpu", trust_remote_code=True)


def _similarity(left: Any, right: Any) -> float:
    import numpy as np

    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def find_duplicates(
    repo_paths: list[str], similarity_threshold: float = 0.85
) -> dict[str, Any]:
    """Find cross-repository function/class pairs above a cosine threshold."""
    if not isinstance(repo_paths, list) or not repo_paths:
        return {"success": False, "error": "repo_paths must be a non-empty list."}
    if not isinstance(similarity_threshold, (int, float)) or not -1 <= similarity_threshold <= 1:
        return {"success": False, "error": "similarity_threshold must be between -1 and 1."}

    roots: list[tuple[Path, str]] = []
    for value in repo_paths:
        if not isinstance(value, str) or not value.strip():
            return {"success": False, "error": "Each repo path must be a non-empty string."}
        path = Path(value).expanduser().resolve()
        if not path.exists():
            return {"success": False, "repo_paths": repo_paths, "error": f"Path does not exist: {value}"}
        roots.append((path, str(path)))

    try:
        chunks: list[_Chunk] = []
        for path, label in roots:
            chunks.extend(_chunks(path, label))
        if len({chunk.repo_path for chunk in chunks}) < 2:
            return {
                "success": True,
                "repo_paths": [str(path) for path, _ in roots],
                "similarity_threshold": float(similarity_threshold),
                "matches": [],
                "chunks": len(chunks),
            }

        vectors = _embedder().encode([chunk.source for chunk in chunks])
        matches: list[dict[str, Any]] = []
        for index, left in enumerate(chunks):
            for right_index in range(index + 1, len(chunks)):
                right = chunks[right_index]
                if left.repo_path == right.repo_path:
                    continue
                score = _similarity(vectors[index], vectors[right_index])
                if score >= similarity_threshold:
                    matches.append(
                        {
                            "similarity": score,
                            "left": {
                                "repo_path": left.repo_path,
                                "file_path": left.file_path,
                                "function_name": left.function_name,
                            },
                            "right": {
                                "repo_path": right.repo_path,
                                "file_path": right.file_path,
                                "function_name": right.function_name,
                            },
                        }
                    )
        matches.sort(key=lambda match: match["similarity"], reverse=True)
        return {
            "success": True,
            "repo_paths": [str(path) for path, _ in roots],
            "similarity_threshold": float(similarity_threshold),
            "matches": matches,
            "chunks": len(chunks),
        }
    except Exception as exc:
        return {"success": False, "repo_paths": repo_paths, "error": str(exc)}


__all__ = ["find_duplicates"]
