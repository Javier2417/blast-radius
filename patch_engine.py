"""Unified diff applier, AST diff classifier, hot __code__ patcher, and overlay loader."""

from __future__ import annotations

import array
import ast
from dataclasses import dataclass, field
import gc
import gzip
import importlib.abc
import importlib.machinery
import importlib.util
import io
import marshal
import math
import os
from pathlib import Path
import re
import sys
import types
import builtins

_DIFF_HDR = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_HDR = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_CODING_RE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")

NARROWABLE_DICTS = frozenset(
    {
        "FUNCTIONS",
        "NO_PAREN_FUNCTIONS",
        "EXPRESSION_PARSERS",
        "STATEMENT_PARSERS",
        "UNARY_PARSERS",
        "STRING_PARSERS",
        "NUMERIC_PARSERS",
        "PRIMARY_PARSERS",
        "PLACEHOLDER_PARSERS",
        "RANGE_PARSERS",
        "PIPE_SYNTAX_TRANSFORM_PARSERS",
        "PROPERTY_PARSERS",
        "CONSTRAINT_PARSERS",
        "ALTER_PARSERS",
        "ALTER_ALTER_PARSERS",
        "NO_PAREN_FUNCTION_PARSERS",
        "FUNCTION_PARSERS",
        "QUERY_MODIFIER_PARSERS",
        "SET_PARSERS",
        "SHOW_PARSERS",
        "TYPE_LITERAL_PARSERS",
        "ANALYZE_EXPRESSION_PARSERS",
        "TRANSFORMS",
        "TYPE_MAPPING",
        "NUMERIC_ESCAPES",
        "PROPERTIES_LOCATION",
        "EXPRESSION_METADATA",
    }
)

LIST_FIXTURES = frozenset(
    {
        "tests/fixtures/identity.sql",
        "tests/fixtures/partial.sql",
        "tests/fixtures/optimizer/qualify_columns__invalid.sql",
    }
)


class PatchError(ValueError):
    """Raised when a patch cannot be parsed or applied cleanly against pinned sources."""


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    # (op: ' ' | '-' | '+', text_without_nl: str, has_nl: bool)
    ops: list[tuple[str, str, bool]]


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)


def _unquote_git_path(path: str) -> str:
    if path.startswith('"') and path.endswith('"'):
        return bytes(path[1:-1], "utf-8").decode("unicode_escape")
    return path


def parse_patch(text: str) -> list[FilePatch]:
    """Parse a git unified diff strictly into FilePatch records."""
    if not text.strip():
        return []
    lines = text.splitlines()
    patches: list[FilePatch] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m = _DIFF_HDR.match(line)
        if not m:
            i += 1
            continue
        old_p: str | None = _unquote_git_path(m.group(1))
        new_p: str | None = _unquote_git_path(m.group(2))
        i += 1
        hunks: list[Hunk] = []

        while i < n and not _DIFF_HDR.match(lines[i]):
            hl = lines[i]
            if hl == "GIT binary patch" or hl.startswith("Binary files "):
                raise PatchError(f"Binary diff unsupported for {old_p or new_p}")
            if hl.startswith("--- "):
                raw = hl[4:].strip()
                old_p = None if raw == "/dev/null" else _unquote_git_path(raw.removeprefix("a/"))
                i += 1
                continue
            if hl.startswith("+++ "):
                raw = hl[4:].strip()
                new_p = None if raw == "/dev/null" else _unquote_git_path(raw.removeprefix("b/"))
                i += 1
                continue
            if hl.startswith(("index ", "old mode ", "new mode ", "deleted file mode ", "new file mode ", "similarity index ", "rename from ", "rename to ", "copy from ", "copy to ")):
                i += 1
                continue

            hm = _HUNK_HDR.match(hl)
            if hm:
                old_start = int(hm.group(1))
                old_count = int(hm.group(2)) if hm.group(2) is not None else 1
                new_start = int(hm.group(3))
                new_count = int(hm.group(4)) if hm.group(4) is not None else 1
                i += 1
                ops: list[tuple[str, str, bool]] = []
                o_seen = 0
                n_seen = 0
                while i < n and (o_seen < old_count or n_seen < new_count):
                    dl = lines[i]
                    if dl.startswith("\\ No newline at end of file"):
                        if not ops:
                            raise PatchError("Orphan no-newline marker")
                        op, txt, _ = ops[-1]
                        ops[-1] = (op, txt, False)
                        i += 1
                        continue
                    if dl == "":
                        # Blank context line where trailing space was trimmed
                        tag, content = " ", ""
                    else:
                        tag, content = dl[0], dl[1:]
                    if tag == " ":
                        o_seen += 1
                        n_seen += 1
                    elif tag == "-":
                        o_seen += 1
                    elif tag == "+":
                        n_seen += 1
                    else:
                        raise PatchError(f"Malformed hunk line in {old_p or new_p}: {dl!r}")
                    ops.append((tag, content, True))
                    i += 1
                # Trailing no-newline marker right after last line of hunk
                if i < n and lines[i].startswith("\\ No newline at end of file"):
                    if not ops:
                        raise PatchError("Orphan trailing no-newline marker")
                    op, txt, _ = ops[-1]
                    ops[-1] = (op, txt, False)
                    i += 1
                if o_seen != old_count or n_seen != new_count:
                    raise PatchError(f"Hunk line count mismatch in {old_p or new_p}")
                hunks.append(Hunk(old_start, old_count, new_start, new_count, ops))
                continue

            i += 1

        patches.append(FilePatch(old_p, new_p, hunks))

    return patches


def apply_patch(source_text: str | None, fp: FilePatch) -> str | None:
    """Strictly apply FilePatch to pinned source_text. Raises PatchError on mismatch."""
    if fp.new_path is None:
        return None
    if fp.old_path is None:
        parts = [txt + ("\n" if has_nl else "") for tag, txt, has_nl in (op for h in fp.hunks for op in h.ops) if tag == "+"]
        return "".join(parts)

    if source_text is None:
        raise PatchError(f"Missing pinned source for {fp.old_path}")

    old_lines = source_text.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0

    for h in fp.hunks:
        start_idx = 0 if h.old_start == 0 else h.old_start - 1
        if start_idx < cursor or start_idx > len(old_lines):
            raise PatchError(f"Invalid hunk start {h.old_start} for {fp.old_path}")
        out.extend(old_lines[cursor:start_idx])
        idx = start_idx
        for tag, txt, has_nl in h.ops:
            expected_line = txt + ("\n" if has_nl else "")
            if tag == " ":
                if idx >= len(old_lines) or old_lines[idx] != expected_line:
                    # Tolerate trailing newline difference only if last line of file without \n matches without it
                    if idx < len(old_lines) and old_lines[idx].rstrip("\n") == txt and not old_lines[idx].endswith("\n"):
                        out.append(old_lines[idx])
                    else:
                        raise PatchError(f"Context mismatch at line {idx + 1} in {fp.old_path}")
                else:
                    out.append(expected_line)
                idx += 1
            elif tag == "-":
                if idx >= len(old_lines) or old_lines[idx].rstrip("\n") != txt:
                    raise PatchError(f"Delete mismatch at line {idx + 1} in {fp.old_path}")
                idx += 1
            elif tag == "+":
                out.append(expected_line)
        cursor = idx

    out.extend(old_lines[cursor:])
    return "".join(out)


def _filter_sql_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))


def _sql_fixture_list(text: str) -> list[str]:
    entries: list[str] = []
    buf: list[str] = []
    for line in _filter_sql_comments(text).splitlines():
        buf.append(line)
        if not line.startswith("#"):
            entries.append("\n".join(buf))
            buf.clear()
    return entries


def _sql_fixture_pairs(text: str) -> list[tuple[str, str]]:
    stmts = _filter_sql_comments(text).split(";")
    pairs: list[tuple[str, str]] = []
    size = len(stmts)
    for i in range(0, size, 2):
        if i + 1 < size:
            pairs.append((stmts[i].strip(), stmts[i + 1].strip()))
    return pairs


def _has_coding_cookie_diff(old_bytes: bytes, new_bytes: bytes) -> bool:
    def extract(b: bytes) -> tuple[bytes | None, ...]:
        res: list[bytes | None] = []
        for line in b.splitlines()[:2]:
            m = _CODING_RE.match(line)
            res.append(m.group(1).lower() if m else None)
        return tuple(res)

    return extract(old_bytes) != extract(new_bytes)


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


PURE_EQUIV_ENUMS = frozenset(
    {
        "ErrorLevel",
        "DType",
        "PropertiesLocation",
        "ScopeType",
        "IsPredicatePrecedence",
        "TrieResult",
    }
)

SPECIAL_EQUIV_ENUMS = frozenset(
    {
        "NormalizationStrategy",
        "UsingColumnOrder",
        "AliasPrecedence",
    }
)


def _is_none_const(node: object) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_enum_attr(node: object, enum_names: frozenset[str]) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    val = node.value
    if isinstance(val, ast.Name):
        return val.id in enum_names
    if isinstance(val, ast.Attribute):
        return val.attr in enum_names
    return False


def _is_safe_special_lhs(node: object) -> bool:
    if isinstance(node, ast.Name):
        return node.id in ("strategy", "column_order")
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == "self" and node.attr in ("normalization_strategy", "USING_COLUMN_ORDER"):
                return True
            if node.value.id == "dialect" and node.attr == "USING_COLUMN_ORDER":
                return True
        return False
    if isinstance(node, ast.Call):
        fn = node.func
        return (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "self"
            and fn.attr == "column_alias_precedence"
        )
    return False


def _compares_semantically_equal(a: ast.Compare, b: ast.Compare) -> bool:
    if len(a.ops) != 1 or len(b.ops) != 1 or len(a.comparators) != 1 or len(b.comparators) != 1:
        return False
    op_pair = {type(a.ops[0]), type(b.ops[0])}
    if op_pair not in ({ast.Is, ast.Eq}, {ast.IsNot, ast.NotEq}):
        return False
    if not (_ast_equal(a.left, b.left, True) and _ast_equal(a.comparators[0], b.comparators[0], True)):
        return False
    lhs, rhs = a.left, a.comparators[0]
    if _is_none_const(rhs) or _is_none_const(lhs):
        return True
    if _is_enum_attr(rhs, PURE_EQUIV_ENUMS) or _is_enum_attr(lhs, PURE_EQUIV_ENUMS):
        return True
    if _is_enum_attr(rhs, SPECIAL_EQUIV_ENUMS) and _is_safe_special_lhs(lhs):
        return True
    if _is_enum_attr(lhs, SPECIAL_EQUIV_ENUMS) and _is_safe_special_lhs(rhs):
        return True
    return False


def _ast_equal(a: object, b: object, allow_equiv: bool = False) -> bool:
    """Fast recursive structural AST equality ignoring source positions."""
    if a is b:
        return True
    if allow_equiv and isinstance(a, ast.Compare) and isinstance(b, ast.Compare):
        if _compares_semantically_equal(a, b):
            return True
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        for f in a._fields:
            if not _ast_equal(getattr(a, f, None), getattr(b, f, None), allow_equiv):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not _ast_equal(x, y, allow_equiv):
                return False
        return True
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):  # type: ignore[arg-type]
            return True
        return repr(a) == repr(b)
    return a == b


def _fn_header_equal(
    f1: ast.FunctionDef | ast.AsyncFunctionDef,
    f2: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return (
        type(f1) is type(f2)
        and f1.name == f2.name
        and _ast_equal(f1.args, f2.args)
        and _ast_equal(f1.decorator_list, f2.decorator_list)
        and _ast_equal(f1.returns, f2.returns)
        and _ast_equal(getattr(f1, "type_params", []), getattr(f2, "type_params", []))
    )


def _cls_header_equal(c1: ast.ClassDef, c2: ast.ClassDef) -> bool:
    return (
        c1.name == c2.name
        and _ast_equal(c1.bases, c2.bases)
        and _ast_equal(c1.keywords, c2.keywords)
        and _ast_equal(c1.decorator_list, c2.decorator_list)
        and _ast_equal(getattr(c1, "type_params", []), getattr(c2, "type_params", []))
    )


def _has_class_def(stmts: list[ast.stmt]) -> bool:
    for s in stmts:
        for sub in ast.walk(s):
            if isinstance(sub, ast.ClassDef):
                return True
    return False


def _add_span(lines: set[int], start: int, end: int | None) -> None:
    e = end if end is not None and end >= start else start
    lines.update(range(start, e + 1))


def _collect_block_diff_lines(
    old_stmts: list[ast.stmt],
    new_stmts: list[ast.stmt],
    parent_header_lines: list[int],
    out_lines: set[int],
) -> None:
    """Recursively collect minimal sound old line numbers covering AST diffs."""
    if len(old_stmts) != len(new_stmts):
        if not old_stmts:
            out_lines.update(parent_header_lines)
            return
        p = 0
        limit = min(len(old_stmts), len(new_stmts))
        while p < limit and _ast_equal(old_stmts[p], new_stmts[p]):
            p += 1
        s = 0
        rem_old = len(old_stmts) - p
        rem_new = len(new_stmts) - p
        max_s = min(rem_old, rem_new)
        while s < max_s and _ast_equal(old_stmts[len(old_stmts) - 1 - s], new_stmts[len(new_stmts) - 1 - s]):
            s += 1
        lo = max(0, p - 1)
        hi = len(old_stmts) - 1 - max(0, s - 1)
        if p == 0:
            out_lines.update(parent_header_lines)
        for idx in range(lo, hi + 1):
            _add_span(out_lines, old_stmts[idx].lineno, old_stmts[idx].end_lineno)
        return

    for old_s, new_s in zip(old_stmts, new_stmts):
        if _ast_equal(old_s, new_s):
            continue
        if type(old_s) is not type(new_s):
            _add_span(out_lines, old_s.lineno, old_s.end_lineno)
            continue

        if isinstance(old_s, ast.If) and isinstance(new_s, ast.If):
            test_same = _ast_equal(old_s.test, new_s.test)
            hdr = list(range(old_s.lineno, (old_s.test.end_lineno or old_s.lineno) + 1))
            if not test_same:
                out_lines.update(hdr)
            _collect_block_diff_lines(old_s.body, new_s.body, hdr, out_lines)
            _collect_block_diff_lines(old_s.orelse, new_s.orelse, hdr, out_lines)
        elif isinstance(old_s, (ast.For, ast.AsyncFor)) and isinstance(new_s, (ast.For, ast.AsyncFor)):
            hdr_same = _ast_equal(old_s.target, new_s.target) and _ast_equal(old_s.iter, new_s.iter)
            hdr_end = old_s.iter.end_lineno or old_s.lineno
            hdr = list(range(old_s.lineno, hdr_end + 1))
            if not hdr_same:
                out_lines.update(hdr)
            _collect_block_diff_lines(old_s.body, new_s.body, hdr, out_lines)
            _collect_block_diff_lines(old_s.orelse, new_s.orelse, hdr, out_lines)
        elif isinstance(old_s, ast.While) and isinstance(new_s, ast.While):
            hdr_same = _ast_equal(old_s.test, new_s.test)
            hdr = list(range(old_s.lineno, (old_s.test.end_lineno or old_s.lineno) + 1))
            if not hdr_same:
                out_lines.update(hdr)
            _collect_block_diff_lines(old_s.body, new_s.body, hdr, out_lines)
            _collect_block_diff_lines(old_s.orelse, new_s.orelse, hdr, out_lines)
        elif isinstance(old_s, (ast.With, ast.AsyncWith)) and isinstance(new_s, (ast.With, ast.AsyncWith)):
            hdr_same = _ast_equal(old_s.items, new_s.items)
            hdr_end = max([it.context_expr.end_lineno or old_s.lineno for it in old_s.items], default=old_s.lineno)
            hdr = list(range(old_s.lineno, hdr_end + 1))
            if not hdr_same:
                out_lines.update(hdr)
            _collect_block_diff_lines(old_s.body, new_s.body, hdr, out_lines)
        elif isinstance(old_s, ast.Try) and isinstance(new_s, ast.Try):
            if len(old_s.handlers) != len(new_s.handlers):
                _add_span(out_lines, old_s.lineno, old_s.end_lineno)
                continue
            hdr = [old_s.lineno]
            _collect_block_diff_lines(old_s.body, new_s.body, hdr, out_lines)
            _collect_block_diff_lines(old_s.orelse, new_s.orelse, hdr, out_lines)
            _collect_block_diff_lines(old_s.finalbody, new_s.finalbody, hdr, out_lines)
            for h1, h2 in zip(old_s.handlers, new_s.handlers):
                if not _ast_equal(h1.type, h2.type) or h1.name != h2.name:
                    _add_span(out_lines, h1.lineno, h1.end_lineno)
                else:
                    _collect_block_diff_lines(h1.body, h2.body, [h1.lineno], out_lines)
        elif isinstance(old_s, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(new_s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _fn_header_equal(old_s, new_s) and not _has_class_def(old_s.body) and not _has_class_def(new_s.body):
                first_l = old_s.body[0].lineno if old_s.body else old_s.lineno
                _collect_block_diff_lines(old_s.body, new_s.body, [first_l], out_lines)
            else:
                out_lines.update(parent_header_lines)
                _add_span(out_lines, old_s.lineno, old_s.end_lineno)
        else:
            _add_span(out_lines, old_s.lineno, old_s.end_lineno)


def _extract_body_only_diffs(
    before: ast.Module,
    after: ast.Module,
    out_lines: set[int],
    hot_pairs: list[tuple[int, ast.FunctionDef | ast.AsyncFunctionDef, ast.FunctionDef | ast.AsyncFunctionDef]],
) -> bool:
    """Return True if all changes are strictly inside function/method bodies."""
    def walk_body(b1: list[ast.stmt], b2: list[ast.stmt]) -> bool:
        if len(b1) != len(b2):
            return False
        for s1, s2 in zip(b1, b2):
            if _ast_equal(s1, s2):
                continue
            if type(s1) is not type(s2):
                return False
            if isinstance(s1, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(s2, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _fn_header_equal(s1, s2):
                    return False
                if _has_class_def(s1.body) or _has_class_def(s2.body):
                    return False
                first_l = s1.body[0].lineno if s1.body else s1.lineno
                _collect_block_diff_lines(s1.body, s2.body, [first_l], out_lines)
                hot_pairs.append((s1.lineno, s1, s2))
            elif isinstance(s1, ast.ClassDef) and isinstance(s2, ast.ClassDef):
                if not _cls_header_equal(s1, s2):
                    return False
                if not walk_body(s1.body, s2.body):
                    return False
            else:
                return False
        return True

    return walk_body(before.body, after.body)


def _ast_key_repr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return f"s:{v}"
        if isinstance(v, int):
            return f"i:{v}"
        return f"r:{v!r}"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "exp":
            return f"t:{node.attr}"
        if node.value.id == "TokenType":
            return f"e:TokenType.{node.attr}"
    if isinstance(node, ast.Tuple):
        parts = [_ast_key_repr(e) for e in node.elts]
        if all(p is not None for p in parts):
            return "u:(" + ",".join(parts) + ")"  # type: ignore[arg-type]
    return None


class MapContext:
    """Pre-indexed view of cov_map.marshal with lazy per-file blob decoding."""

    def __init__(self, cov_map: dict):
        self.cov_map = cov_map
        self.ids: list[str] = cov_map["ids"]
        self.cpu_ms: list[int] = cov_map["cpu_ms"]
        self._g_data = array.array("H")
        self._g_data.frombytes(cov_map["groups_data"])
        self._g_offs = array.array("I")
        self._g_offs.frombytes(cov_map["groups_offs"])
        self._eg_data = array.array("H")
        self._eg_data.frombytes(cov_map["eg_data"])
        self._eg_offs = array.array("I")
        self._eg_offs.frombytes(cov_map["eg_offs"])
        self._er_data = array.array("i")
        self._er_data.frombytes(cov_map["er_data"])
        self._er_offs = array.array("I")
        self._er_offs.frombytes(cov_map["er_offs"])
        self.reads: dict[str, int] = cov_map["reads"]
        self.dialects: dict[str, int] = cov_map["dialects"]
        self.exprs: dict[str, int] = cov_map["exprs"]
        self.file_imports: dict[str, list[str]] = cov_map["file_imports"]
        self.file_all_tests: dict[str, int] = cov_map["file_all_tests"]
        self._lines_blobs: dict[str, bytes] = cov_map["lines_blobs"]
        self._entry_blobs: dict[str, bytes] = cov_map["entry_blobs"]
        self._dkey_blobs: dict[str, bytes] = cov_map["dict_key_blobs"]
        self.attr_refs_sqlglot: dict[str, list[tuple[str, int]]] = cov_map.get("attr_refs_sqlglot", {})
        self.attr_refs_tests: dict[str, list[str]] = cov_map.get("attr_refs_tests", {})
        self.metaclass_attrs: set[str] = set(cov_map.get("metaclass_attrs", ()))
        self._lines_cache: dict[str, dict[int, int]] = {}
        self._entry_cache: dict[str, dict[int, int]] = {}
        self._dkey_cache: dict[str, dict[tuple[str, str, str], int]] = {}

        self.by_test_file: dict[str, list[int]] = {}
        self.id_to_idx: dict[str, int] = {}
        for idx, tid in enumerate(self.ids):
            self.id_to_idx[tid] = idx
            tfile = tid.split("::", 1)[0]
            self.by_test_file.setdefault(tfile, []).append(idx)
        self.all_idx: set[int] = set(range(len(self.ids)))
        self.dialect_test_indices: set[int] = {
            idx
            for tf, idxs in self.by_test_file.items()
            if tf.startswith("tests/dialects/")
            for idx in idxs
        }

    def get_group(self, gid: int) -> array.array:
        return self._g_data[self._g_offs[gid] : self._g_offs[gid + 1]]

    def get_entry_group(self, egid: int) -> array.array:
        return self._eg_data[self._eg_offs[egid] : self._eg_offs[egid + 1]]

    def get_entry_row(self, rid: int) -> list[tuple[int, int]]:
        sl = self._er_data[self._er_offs[rid] : self._er_offs[rid + 1]]
        return [(sl[i], sl[i + 1]) for i in range(0, len(sl), 2)]

    def get_lines(self, rel_path: str) -> dict[int, int]:
        res = self._lines_cache.get(rel_path)
        if res is None:
            blob = self._lines_blobs.get(rel_path)
            res = marshal.loads(blob) if blob is not None else {}
            self._lines_cache[rel_path] = res
        return res

    def get_entries(self, rel_path: str) -> dict[int, int]:
        res = self._entry_cache.get(rel_path)
        if res is None:
            blob = self._entry_blobs.get(rel_path)
            res = marshal.loads(blob) if blob is not None else {}
            self._entry_cache[rel_path] = res
        return res

    def get_dkeys(self, rel_path: str) -> dict[tuple[str, str, str], int]:
        res = self._dkey_cache.get(rel_path)
        if res is None:
            blob = self._dkey_blobs.get(rel_path)
            res = marshal.loads(blob) if blob is not None else {}
            self._dkey_cache[rel_path] = res
        return res


def _try_dict_edit(
    rel_path: str,
    before: ast.Module,
    after: ast.Module,
    ctx: MapContext,
) -> set[int] | None:
    """If edits are strictly values inside existing tracked class dicts, return candidate test indices."""
    dkey_map = ctx.get_dkeys(rel_path)

    def check_body(cls_name: str, b1: list[ast.stmt], b2: list[ast.stmt], out: set[int]) -> bool:
        if len(b1) != len(b2):
            return False
        for s1, s2 in zip(b1, b2):
            if _ast_equal(s1, s2):
                continue
            if type(s1) is not type(s2):
                return False
            if isinstance(s1, ast.ClassDef) and isinstance(s2, ast.ClassDef):
                if not _cls_header_equal(s1, s2):
                    return False
                if not check_body(s1.name, s1.body, s2.body, out):
                    return False
                continue
            attr_name: str | None = None
            d1: ast.AST | None = None
            d2: ast.AST | None = None
            if isinstance(s1, ast.Assign) and isinstance(s2, ast.Assign):
                if len(s1.targets) == 1 and len(s2.targets) == 1 and isinstance(s1.targets[0], ast.Name) and isinstance(s2.targets[0], ast.Name):
                    if s1.targets[0].id == s2.targets[0].id:
                        attr_name = s1.targets[0].id
                        d1, d2 = s1.value, s2.value
            elif isinstance(s1, ast.AnnAssign) and isinstance(s2, ast.AnnAssign):
                if isinstance(s1.target, ast.Name) and isinstance(s2.target, ast.Name) and s1.target.id == s2.target.id:
                    attr_name = s1.target.id
                    d1, d2 = s1.value, s2.value
            if not attr_name or attr_name not in NARROWABLE_DICTS:
                return False
            if not isinstance(d1, ast.Dict) or not isinstance(d2, ast.Dict):
                return False
            if not cls_name:
                return False
            # Ensure unpacks (**Base.DICT) match identically
            u1 = [v for k, v in zip(d1.keys, d1.values) if k is None]
            u2 = [v for k, v in zip(d2.keys, d2.values) if k is None]
            if not _ast_equal(u1, u2):
                return False
            kv1: dict[str, ast.AST] = {}
            kv2: dict[str, ast.AST] = {}
            for k, v in zip(d1.keys, d1.values):
                if k is None:
                    continue
                kr = _ast_key_repr(k)
                if kr is None:
                    return False
                kv1[kr] = v
            for k, v in zip(d2.keys, d2.values):
                if k is None:
                    continue
                kr = _ast_key_repr(k)
                if kr is None:
                    return False
                kv2[kr] = v
            if set(kv1) != set(kv2):
                return False
            changed_keys = [kr for kr in kv1 if not _ast_equal(kv1[kr], kv2[kr])]
            if not changed_keys:
                return False
            for kr in changed_keys:
                gid = dkey_map.get((cls_name, attr_name, kr))
                if gid is not None:
                    out.update(ctx.get_group(gid))
                star_gid = dkey_map.get((cls_name, attr_name, "*"))
                if star_gid is not None:
                    out.update(ctx.get_group(star_gid))
        return True

    selected: set[int] = set()
    if check_body("", before.body, after.body, selected):
        return selected
    return None


def _is_pure_const_expr(node: object) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        return _is_pure_const_expr(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_pure_const_expr(el) for el in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (k is None or _is_pure_const_expr(k)) and _is_pure_const_expr(v)
            for k, v in zip(node.keys, node.values)
        )
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _is_pure_const_expr(node.value)
    return False


def _try_class_attr_edit(
    before: ast.Module,
    after: ast.Module,
    ctx: MapContext,
) -> list[tuple[tuple[str, ...], str, bool, object]] | None:
    """If edits strictly change RHS of pure existing ClassDef attributes, return (cls_path, attr, has_lit, lit_val)."""
    changed_attrs: list[tuple[tuple[str, ...], str, bool, object]] = []

    def check_body(cls_path: tuple[str, ...], b1: list[ast.stmt], b2: list[ast.stmt]) -> bool:
        if len(b1) != len(b2):
            return False
        for s1, s2 in zip(b1, b2):
            if _ast_equal(s1, s2):
                continue
            if type(s1) is not type(s2):
                return False
            if isinstance(s1, ast.ClassDef) and isinstance(s2, ast.ClassDef):
                if not _cls_header_equal(s1, s2):
                    return False
                if not check_body(cls_path + (s1.name,), s1.body, s2.body):
                    return False
                continue
            if not cls_path:
                return False
            attr_name: str | None = None
            v1: ast.AST | None = None
            v2: ast.AST | None = None
            if isinstance(s1, ast.Assign) and isinstance(s2, ast.Assign):
                if (
                    len(s1.targets) == 1
                    and len(s2.targets) == 1
                    and isinstance(s1.targets[0], ast.Name)
                    and isinstance(s2.targets[0], ast.Name)
                    and s1.targets[0].id == s2.targets[0].id
                ):
                    attr_name = s1.targets[0].id
                    v1, v2 = s1.value, s2.value
            elif isinstance(s1, ast.AnnAssign) and isinstance(s2, ast.AnnAssign):
                if (
                    isinstance(s1.target, ast.Name)
                    and isinstance(s2.target, ast.Name)
                    and s1.target.id == s2.target.id
                    and _ast_equal(s1.annotation, s2.annotation)
                ):
                    attr_name = s1.target.id
                    v1, v2 = s1.value, s2.value
            if (
                not attr_name
                or v1 is None
                or v2 is None
                or attr_name in ctx.metaclass_attrs
                or attr_name not in ctx.attr_refs_sqlglot
            ):
                return False
            if not (_is_pure_const_expr(v1) and _is_pure_const_expr(v2)):
                return False
            try:
                lit_val = ast.literal_eval(v2)
                has_lit = True
            except Exception:
                lit_val = None
                has_lit = False
            changed_attrs.append((cls_path, attr_name, has_lit, lit_val))
        return True

    if check_body((), before.body, after.body) and changed_attrs:
        return changed_attrs
    return None


@dataclass
class CommitPlan:
    modified_files: dict[str, str | None]
    can_hot_patch: bool
    hot_targets: dict[str, list[tuple[int, str, str]]]
    new_sources_for_hot: dict[str, str]
    candidate_indices: set[int]
    entry_subsets: dict[int, set[int] | None]
    has_syntax_error: bool
    dialect_scope: set[str] | None
    kind: str
    hot_cls_attrs: dict[str, list[tuple[tuple[str, ...], str, object]]] = field(default_factory=dict)
    max_nodes_touched: bool = False


def _extract_entry_subsets_for_lines(
    rel_path: str,
    old_lines: set[int],
    ctx: MapContext,
    out_candidates: set[int],
    out_subsets: dict[int, set[int] | None],
) -> None:
    """Resolve line coverage + exact entry subset info for changed old lines."""
    lines_map = ctx.get_lines(rel_path)
    if not lines_map:
        return
    entry_cov = ctx.get_entries(rel_path)

    for ln in old_lines:
        gid = lines_map.get(ln)
        if gid is None:
            continue
        tlist = ctx.get_group(gid)
        out_candidates.update(tlist)
        erow_id = entry_cov.get(ln)
        row_map: dict[int, int] = dict(ctx.get_entry_row(erow_id)) if erow_id is not None else {}
        for tidx in tlist:
            egid = row_map.get(tidx)
            if egid is None or egid == -1:
                out_subsets[tidx] = None
            else:
                cur = out_subsets.get(tidx)
                if cur is None and tidx in out_subsets:
                    continue
                if cur is None:
                    cur = set()
                    out_subsets[tidx] = cur
                cur.update(ctx.get_entry_group(egid))


def _find_modified_expr_classes(before: ast.Module, after: ast.Module) -> list[str] | None:
    """If all diffs are inside named top-level ClassDefs, return their names."""
    if len(before.body) != len(after.body):
        return None
    names: list[str] = []
    for s1, s2 in zip(before.body, after.body):
        if _ast_equal(s1, s2):
            continue
        if isinstance(s1, ast.ClassDef) and isinstance(s2, ast.ClassDef) and s1.name == s2.name:
            names.append(s1.name)
        else:
            return None
    return names if names else None


def analyze_commit(patch_text: str, root_dir: Path, ctx_or_map: MapContext | dict) -> CommitPlan:
    """Analyze unified diff against pinned tree using AST + coverage maps."""
    ctx = ctx_or_map if isinstance(ctx_or_map, MapContext) else MapContext(ctx_or_map)
    file_patches = parse_patch(patch_text)
    if len({fp.old_path or fp.new_path for fp in file_patches}) != len(file_patches):
        raise PatchError("Repeated file diff")

    ids = ctx.ids
    reads = ctx.reads
    dialects = ctx.dialects
    exprs = ctx.exprs
    file_imports = ctx.file_imports
    by_test_file = ctx.by_test_file
    id_to_idx = ctx.id_to_idx
    all_idx = ctx.all_idx

    modified_files: dict[str, str | None] = {}
    candidate_indices: set[int] = set()
    entry_subsets: dict[int, set[int] | None] = {}
    hot_targets: dict[str, list[tuple[int, str, str]]] = {}
    hot_cls_attrs: dict[str, list[tuple[tuple[str, ...], str, object]]] = {}
    new_sources_for_hot: dict[str, str] = {}
    can_hot_patch = True
    allow_subsets = True
    has_syntax_error = False
    dialect_scope: set[str] | None = set()
    reasons: list[str] = []

    def invalidate_subsets(disable_hot: bool = True) -> None:
        nonlocal can_hot_patch, allow_subsets
        if disable_hot:
            can_hot_patch = False
        allow_subsets = False
        entry_subsets.clear()

    for fp in file_patches:
        rel_path = fp.old_path or fp.new_path
        assert rel_path is not None
        disk_path = root_dir / rel_path
        old_text = disk_path.read_text(encoding="utf-8") if (fp.old_path and disk_path.is_file()) else None
        new_text = apply_patch(old_text, fp)

        if fp.old_path != fp.new_path or old_text is None or new_text is None:
            modified_files[rel_path] = new_text
            if rel_path in file_imports:
                for tf in file_imports[rel_path]:
                    candidate_indices.update(by_test_file.get(tf, ()))
            else:
                candidate_indices.update(all_idx)
            invalidate_subsets()
            dialect_scope = None
            reasons.append("structural_file")
            continue

        # 1. Fixture files
        if not rel_path.endswith(".py"):
            modified_files[rel_path] = new_text
            read_gid = reads.get(rel_path)
            read_tidxs = ctx.get_group(read_gid) if read_gid is not None else list(all_idx)

            if rel_path.endswith(".sql") and rel_path.startswith("tests/fixtures/"):
                if rel_path in LIST_FIXTURES:
                    e1 = _sql_fixture_list(old_text)
                    e2 = _sql_fixture_list(new_text)
                else:
                    e1 = _sql_fixture_pairs(old_text)  # type: ignore[assignment]
                    e2 = _sql_fixture_pairs(new_text)  # type: ignore[assignment]
                if e1 == e2:
                    reasons.append(f"fixture_noop:{rel_path}")
                    continue
                if len(e1) == len(e2):
                    changed_idxs = {i for i, (a, b) in enumerate(zip(e1, e2)) if a != b}
                    candidate_indices.update(read_tidxs)
                    if allow_subsets:
                        for tidx in read_tidxs:
                            prev = entry_subsets.get(tidx)
                            if prev is None and tidx in entry_subsets:
                                continue
                            if prev is None:
                                entry_subsets[tidx] = set(changed_idxs)
                            else:
                                prev.update(changed_idxs)
                    dialect_scope = None
                    reasons.append(f"fixture_entry:{rel_path}")
                    continue

            candidate_indices.update(read_tidxs)
            invalidate_subsets(disable_hot=False)
            dialect_scope = None
            reasons.append(f"fixture_full:{rel_path}")
            continue

        # 2. Python files: check syntax & AST equality
        old_bytes = old_text.encode("utf-8")
        new_bytes = new_text.encode("utf-8")
        try:
            after_tree = compile(new_bytes, rel_path, "exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True)
            compile(after_tree, rel_path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            modified_files[rel_path] = new_text
            has_syntax_error = True
            invalidate_subsets()
            dialect_scope = None
            if rel_path in file_imports:
                for tf in file_imports[rel_path]:
                    candidate_indices.update(by_test_file.get(tf, ()))
            if rel_path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                stem = Path(rel_path).stem
                if stem in dialects:
                    candidate_indices.update(ctx.get_group(dialects[stem]))
                for meta_t in ("tests/test_dialect_imports.py", "tests/test_dialect_entry_points.py", "tests/dialects/test_dialect.py"):
                    candidate_indices.update(by_test_file.get(meta_t, ()))
            if not candidate_indices:
                candidate_indices.update(all_idx)
            reasons.append(f"syntax_err:{rel_path}")
            continue

        before_tree = ast.parse(old_text, filename=rel_path)
        if not _has_coding_cookie_diff(old_bytes, new_bytes):
            if _ast_equal(before_tree, after_tree):
                reasons.append(f"ast_noop:{rel_path}")
                continue
            if rel_path.startswith("sqlglot/") and rel_path != "sqlglot/executor/env.py":
                if _ast_equal(before_tree, after_tree, allow_equiv=True):
                    reasons.append(f"ast_equiv:{rel_path}")
                    continue

        modified_files[rel_path] = new_text

        if rel_path in reads:
            candidate_indices.update(ctx.get_group(reads[rel_path]))

        # Special test files
        if rel_path.startswith("tests/"):
            invalidate_subsets()
            dialect_scope = None
            if rel_path == "tests/dialects/test_dialect.py":
                candidate_indices.update(ctx.dialect_test_indices)
            elif Path(rel_path).name.startswith("test_"):
                body_lines: set[int] = set()
                hp: list = []
                if _extract_body_only_diffs(before_tree, after_tree, body_lines, hp):
                    if hp and all(fn_old.name.startswith("test_") for _, fn_old, _ in hp):
                        changed_names = {fn_old.name for _, fn_old, _ in hp}
                        for tidx in by_test_file.get(rel_path, ()):
                            mth = ids[tidx].rsplit("::", 1)[1]
                            if mth in changed_names:
                                candidate_indices.add(tidx)
                        reasons.append(f"test_method:{rel_path}")
                        continue
                candidate_indices.update(by_test_file.get(rel_path, ()))
            else:
                for tf in file_imports.get(rel_path, ()):
                    candidate_indices.update(by_test_file.get(tf, ()))
            reasons.append(f"test_file:{rel_path}")
            continue

        # 3. Function/method body-only edits
        diff_lines: set[int] = set()
        hp_list: list = []
        if _extract_body_only_diffs(before_tree, after_tree, diff_lines, hp_list):
            if allow_subsets:
                _extract_entry_subsets_for_lines(rel_path, diff_lines, ctx, candidate_indices, entry_subsets)
            else:
                _extract_entry_subsets_for_lines(rel_path, diff_lines, ctx, candidate_indices, {})
            if can_hot_patch:
                new_sources_for_hot[rel_path] = new_text
                hot_targets[rel_path] = [(lineno, fn1.name, fn2.name) for lineno, fn1, fn2 in hp_list]
            reasons.append(f"body:{rel_path}({len(diff_lines)}L)")
            continue

        # 4. Tracked dict value edit
        dict_cands = _try_dict_edit(rel_path, before_tree, after_tree, ctx)
        if dict_cands is not None:
            can_hot_patch = False
            candidate_indices.update(dict_cands)
            for tidx in dict_cands:
                entry_subsets[tidx] = None
            if rel_path.startswith("sqlglot/typing/"):
                for tidx in by_test_file.get("tests/test_typing.py", ()):
                    candidate_indices.add(tidx)
                    entry_subsets[tidx] = None
            if dialect_scope is not None:
                dialect_scope.add(Path(rel_path).stem)
            reasons.append(f"dict:{rel_path}")
            continue

        # 4b. Class-attribute dataflow narrowing for Dialect/Parser/Generator/Typing files
        if rel_path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
            stem = Path(rel_path).stem
            if stem not in ("dialect", "base", "__init__"):
                changed_attrs = _try_class_attr_edit(before_tree, after_tree, ctx)
                if changed_attrs is not None:
                    dialect_set = set(ctx.get_group(dialects[stem])) if stem in dialects else None
                    for _cls_path, attr, _has_lit, _lit_val in changed_attrs:
                        attr_cands: set[int] = set()
                        attr_subs: dict[int, set[int] | None] = {}
                        by_ref_file: dict[str, set[int]] = {}
                        for ref_rel, ln in ctx.attr_refs_sqlglot[attr]:
                            by_ref_file.setdefault(ref_rel, set()).add(ln)
                        for ref_rel, lns in by_ref_file.items():
                            _extract_entry_subsets_for_lines(ref_rel, lns, ctx, attr_cands, attr_subs)
                        if dialect_set is not None:
                            attr_cands &= dialect_set
                        candidate_indices.update(attr_cands)
                        if allow_subsets:
                            for tidx in attr_cands:
                                sub = attr_subs.get(tidx)
                                if tidx in entry_subsets:
                                    cur = entry_subsets[tidx]
                                    if cur is not None:
                                        if sub is None:
                                            entry_subsets[tidx] = None
                                        else:
                                            cur.update(sub)
                                else:
                                    entry_subsets[tidx] = None if sub is None else set(sub)
                        for tf in ctx.attr_refs_tests.get(attr, ()):
                            for tidx in by_test_file.get(tf, ()):
                                candidate_indices.add(tidx)
                                entry_subsets[tidx] = None
                    if rel_path.startswith("sqlglot/typing/"):
                        for tidx in by_test_file.get("tests/test_typing.py", ()):
                            candidate_indices.add(tidx)
                            entry_subsets[tidx] = None
                    if dialect_scope is not None:
                        dialect_scope.add(stem)
                    if can_hot_patch and all(has_lit for _, _, has_lit, _ in changed_attrs):
                        hot_cls_attrs[rel_path] = [
                            (cls_path, attr, lit_val)
                            for cls_path, attr, _has_lit, lit_val in changed_attrs
                        ]
                    else:
                        can_hot_patch = False
                    reasons.append(f"cls_attr:{rel_path}")
                    continue

        invalidate_subsets()

        # 5. Single Exp class edit in sqlglot/expressions/
        if rel_path.startswith("sqlglot/expressions/") and Path(rel_path).stem != "__init__":
            cls_names = _find_modified_expr_classes(before_tree, after_tree)
            if cls_names is not None:
                for cname in cls_names:
                    if cname in exprs:
                        candidate_indices.update(ctx.get_group(exprs[cname]))
                cls_lines: set[int] = set()
                for s1 in before_tree.body:
                    if isinstance(s1, ast.ClassDef) and s1.name in cls_names:
                        _add_span(cls_lines, s1.lineno, s1.end_lineno)
                _extract_entry_subsets_for_lines(rel_path, cls_lines, ctx, candidate_indices, {})
                for refl_id in (
                    "tests/test_optimizer.py::TestOptimizer::test_gen",
                    "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata",
                ):
                    r_idx = id_to_idx.get(refl_id)
                    if r_idx is not None:
                        candidate_indices.add(r_idx)
                dialect_scope = None
                reasons.append(f"expr:{rel_path}:{cls_names}")
                continue

        # 6. Non-base Dialect/Parser/Generator/Typing file class edit
        if rel_path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
            stem = Path(rel_path).stem
            if stem not in ("dialect", "base", "__init__"):
                if stem in dialects:
                    candidate_indices.update(ctx.get_group(dialects[stem]))
                if dialect_scope is not None:
                    dialect_scope.add(stem)
                for meta_t in ("tests/test_dialect_imports.py", "tests/test_dialect_entry_points.py"):
                    candidate_indices.update(by_test_file.get(meta_t, ()))
                if rel_path.startswith("sqlglot/typing/"):
                    candidate_indices.update(by_test_file.get("tests/test_typing.py", ()))
                mem_idx = id_to_idx.get("tests/test_generator.py::TestGenerator::test_generator_memory")
                if mem_idx is not None:
                    candidate_indices.add(mem_idx)
                reasons.append(f"dialect:{rel_path}")
                continue

        # 7. Broad / module-level fallback
        dialect_scope = None
        fat_gid = ctx.file_all_tests.get(rel_path)
        if fat_gid is not None:
            candidate_indices.update(ctx.get_group(fat_gid))
        for tf in file_imports.get(rel_path, ()):
            candidate_indices.update(by_test_file.get(tf, ()))
        if rel_path in ("sqlglot/dialects/dialect.py", "sqlglot/__init__.py", "sqlglot/typing/__init__.py"):
            candidate_indices.update(ctx.dialect_test_indices)
        if not candidate_indices:
            candidate_indices.update(all_idx)
        reasons.append(f"broad:{rel_path}")

    if not allow_subsets:
        entry_subsets.clear()

    max_nodes_tokens = (
        "max_nodes",
        "_node_count",
        "validate_expression",
        "1_000_000",
        "1000000",
        "50_000",
        "50000",
    )
    max_nodes_touched = "tests/test_parser.py" in modified_files or any(
        line.startswith(("+", "-"))
        and not line.startswith(("+++", "---"))
        and any(tok in line for tok in max_nodes_tokens)
        for line in patch_text.splitlines()
    )

    return CommitPlan(
        modified_files=modified_files,
        can_hot_patch=can_hot_patch and bool(hot_targets or hot_cls_attrs),
        hot_targets=hot_targets,
        new_sources_for_hot=new_sources_for_hot,
        candidate_indices=candidate_indices,
        entry_subsets=entry_subsets,
        has_syntax_error=has_syntax_error,
        dialect_scope=dialect_scope if (dialect_scope and len(dialect_scope) > 0) else None,
        kind="+".join(reasons) or "noop",
        hot_cls_attrs=hot_cls_attrs,
        max_nodes_touched=max_nodes_touched,
    )


def _walk_code_objects(code: types.CodeType, out: list[types.CodeType]) -> None:
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out.append(c)
            _walk_code_objects(c, out)


def _collect_func_nodes(tree: ast.AST, out: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda]) -> None:
    """Collect callable AST nodes in compilation preorder (functions + lambdas)."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            out.append(node)
        _collect_func_nodes(node, out)


def _code_matches(a: types.CodeType, b: types.CodeType) -> bool:
    important_flags = (
        0x04  # CO_VARARGS
        | 0x08  # CO_VARKEYWORDS
        | 0x20  # CO_GENERATOR
        | 0x80  # CO_COROUTINE
        | 0x200  # CO_ASYNC_GENERATOR
    )
    return (
        a.co_name == b.co_name
        and a.co_argcount == b.co_argcount
        and a.co_posonlyargcount == b.co_posonlyargcount
        and a.co_kwonlyargcount == b.co_kwonlyargcount
        and a.co_freevars == b.co_freevars
        and a.co_cellvars == b.co_cellvars
        and (a.co_flags & important_flags) == (b.co_flags & important_flags)
    )


def build_func_index(root_dir: Path) -> dict[tuple[str, int], list[types.FunctionType]]:
    """Index live functions under root_dir by (rel_path, firstlineno) via one gc scan."""
    prefix = str(root_dir.resolve()) + "/"
    idx: dict[tuple[str, int], list[types.FunctionType]] = {}
    for obj in gc.get_objects():
        if isinstance(obj, types.FunctionType):
            co = obj.__code__
            fn = co.co_filename
            if fn.startswith(prefix):
                key = (fn[len(prefix) :], co.co_firstlineno)
                lst = idx.get(key)
                if lst is None:
                    idx[key] = [obj]
                else:
                    lst.append(obj)
    return idx


def _pair_callable_codes(
    old_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    new_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    old_root_code: types.CodeType,
    new_root_code: types.CodeType,
    out_pairs: list[tuple[int, int, types.CodeType]],
) -> bool:
    """Align old and new code objects for an edited outer function and all nested defs."""
    old_codes_all: list[types.CodeType] = []
    new_codes_all: list[types.CodeType] = []
    _walk_code_objects(old_root_code, old_codes_all)
    _walk_code_objects(new_root_code, new_codes_all)

    old_lines_ok = {old_fn.lineno, *(d.lineno for d in old_fn.decorator_list)}
    new_lines_ok = {new_fn.lineno, *(d.lineno for d in new_fn.decorator_list)}
    old_top = [c for c in old_codes_all if c.co_firstlineno in old_lines_ok and c.co_name == old_fn.name]
    new_top = [c for c in new_codes_all if c.co_firstlineno in new_lines_ok and c.co_name == new_fn.name]
    if len(old_top) != 1 or len(new_top) != 1:
        return False

    def match_rec(c_old: types.CodeType, c_new: types.CodeType, depth: int) -> bool:
        if not _code_matches(c_old, c_new):
            return False
        out_pairs.append((depth, c_old.co_firstlineno, c_new))
        subs_old = [x for x in c_old.co_consts if isinstance(x, types.CodeType)]
        subs_new = [x for x in c_new.co_consts if isinstance(x, types.CodeType)]
        if len(subs_old) != len(subs_new):
            return False
        for so, sn in zip(subs_old, subs_new):
            if not match_rec(so, sn, depth + 1):
                return False
        return True

    return match_rec(old_top[0], new_top[0], 0)


def hot_patch_code(
    plan: CommitPlan,
    root_dir: Path,
    func_index: dict[tuple[str, int], list[types.FunctionType]],
) -> bool:
    """Swap __code__ objects in place for BODY_EDITs. Returns False if re-import is required."""
    if not plan.can_hot_patch:
        return False

    swaps: list[tuple[types.FunctionType, types.CodeType]] = []
    for rel_path, new_src in plan.new_sources_for_hot.items():
        old_path = root_dir / rel_path
        old_src = old_path.read_text(encoding="utf-8")
        try:
            old_tree = ast.parse(old_src, filename=str(old_path))
            new_tree = ast.parse(new_src, filename=str(old_path))
            old_mod_code = compile(old_tree, str(old_path), "exec", dont_inherit=True)
            new_mod_code = compile(new_tree, str(old_path), "exec", dont_inherit=True)
        except Exception:
            return False

        # Walk top-level + class functions to find matching old/new AST nodes
        old_funcs: dict[tuple[int, str], ast.FunctionDef | ast.AsyncFunctionDef] = {}
        new_funcs_by_path: dict[tuple[str, ...], tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.FunctionDef | ast.AsyncFunctionDef]] = {}

        def index_fns(
            b1: list[ast.stmt],
            b2: list[ast.stmt],
            prefix: tuple[str, ...],
        ) -> bool:
            if len(b1) != len(b2):
                return False
            for s1, s2 in zip(b1, b2):
                if isinstance(s1, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(s2, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _ast_equal(s1, s2):
                        new_funcs_by_path[prefix + (s1.name,)] = (s1, s2)
                        old_funcs[(s1.lineno, s1.name)] = s1
                elif isinstance(s1, ast.ClassDef) and isinstance(s2, ast.ClassDef):
                    if not index_fns(s1.body, s2.body, prefix + (s1.name,)):
                        return False
            return True

        if not index_fns(old_tree.body, new_tree.body, ()):
            return False

        code_pairs: list[tuple[int, int, types.CodeType]] = []
        for old_fn, new_fn in new_funcs_by_path.values():
            if not _pair_callable_codes(old_fn, new_fn, old_mod_code, new_mod_code, code_pairs):
                return False

        for depth, old_lineno, new_c in code_pairs:
            live_fns = func_index.get((rel_path, old_lineno))
            if not live_fns:
                if depth == 0:
                    return False
                continue
            matched = False
            for fn_obj in live_fns:
                if fn_obj.__code__.co_name != new_c.co_name:
                    if depth == 0:
                        return False
                    continue
                swaps.append((fn_obj, new_c))
                matched = True
            if depth == 0 and not matched:
                return False

    attr_updates: list[tuple[type, str, object]] = []
    for rel_path, items in plan.hot_cls_attrs.items():
        if not rel_path.endswith(".py"):
            return False
        mod = sys.modules.get(rel_path[:-3].replace("/", "."))
        if mod is None:
            return False
        for cls_path, attr_name, literal_val in items:
            obj: object = mod
            for cname in cls_path:
                obj = getattr(obj, cname, None)
                if not isinstance(obj, type):
                    return False
            if attr_name not in obj.__dict__:
                return False
            attr_updates.append((obj, attr_name, literal_val))

    for fn_obj, new_c in swaps:
        fn_obj.__code__ = new_c
    for target_cls, attr_name, literal_val in attr_updates:
        setattr(target_cls, attr_name, literal_val)
    return True


class OverlayLoader(importlib.abc.SourceLoader):
    """SourceLoader serving patched sources from memory and pinned sources from CODE_CACHE."""

    def __init__(self, fullname: str, origin: str, rel_path: str, patched_bytes: bytes | None, code_cache: dict[str, types.CodeType]):
        self.fullname = fullname
        self.origin = origin
        self.rel_path = rel_path
        self.patched_bytes = patched_bytes
        self.code_cache = code_cache

    def get_filename(self, fullname: str) -> str:
        return self.origin

    def get_data(self, path: str) -> bytes:
        if self.patched_bytes is not None:
            return self.patched_bytes
        with open(path, "rb") as f:
            return f.read()

    def get_code(self, fullname: str) -> types.CodeType | None:
        if self.patched_bytes is not None:
            return compile(self.patched_bytes, self.origin, "exec", dont_inherit=True)
        cached = self.code_cache.get(self.origin)
        if cached is not None:
            return cached
        code = super().get_code(fullname)
        if code is not None:
            self.code_cache[self.origin] = code
        return code


class OverlayMetaPathFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder for sqlglot & tests modules supporting deleted and patched files."""

    def __init__(self, root_dir: Path, modified_files: dict[str, str | None], code_cache: dict[str, types.CodeType]):
        self.root_dir = root_dir.resolve()
        self.modified_files = modified_files
        self.code_cache = code_cache

    def find_spec(self, fullname: str, path, target=None):
        top = fullname.partition(".")[0]
        if top not in ("sqlglot", "tests"):
            return None
        parts = fullname.split(".")
        rel_pkg = os.path.join(*parts, "__init__.py")
        rel_mod = os.path.join(*parts) + ".py"

        for is_pkg, rel in ((True, rel_pkg), (False, rel_mod)):
            if rel in self.modified_files:
                content = self.modified_files[rel]
                if content is None:
                    # Deleted file
                    continue
                origin = str(self.root_dir / rel)
                loader = OverlayLoader(fullname, origin, rel, content.encode("utf-8"), self.code_cache)
                spec = importlib.util.spec_from_loader(fullname, loader, origin=origin, is_package=is_pkg)
                if spec is not None:
                    spec.has_location = True
                    if is_pkg:
                        spec.submodule_search_locations = [str((self.root_dir / rel).parent)]
                return spec

            disk_file = self.root_dir / rel
            if disk_file.is_file():
                origin = str(disk_file)
                loader = OverlayLoader(fullname, origin, rel, None, self.code_cache)
                spec = importlib.util.spec_from_loader(fullname, loader, origin=origin, is_package=is_pkg)
                if spec is not None:
                    spec.has_location = True
                    if is_pkg:
                        spec.submodule_search_locations = [str(disk_file.parent)]
                return spec

        return None


def install_overlay(root_dir: Path, modified_files: dict[str, str | None], code_cache: dict[str, types.CodeType]) -> None:
    """Evict loaded sqlglot/tests modules when Python sources change and install file-read hooks."""
    has_py_mod = any(rel.endswith(".py") for rel in modified_files)
    if has_py_mod:
        for name in list(sys.modules):
            top = name.partition(".")[0]
            if top in ("sqlglot", "tests"):
                sys.modules.pop(name, None)

        finder = OverlayMetaPathFinder(root_dir, modified_files, code_cache)
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, OverlayMetaPathFinder)]
        sys.meta_path.insert(0, finder)
        importlib.invalidate_caches()

    # Hook file reads so patched fixtures, typing modules, etc. are read from memory
    prefix = str(root_dir.resolve()) + "/"
    mem_bytes: dict[str, bytes] = {}
    for rel, txt in modified_files.items():
        if txt is not None:
            mem_bytes[rel] = txt.encode("utf-8")

    deleted = {rel for rel, txt in modified_files.items() if txt is None}
    if not mem_bytes and not deleted:
        return

    def resolve_rel(file_obj: object) -> str | None:
        if not isinstance(file_obj, (str, os.PathLike)):
            return None
        s = os.fspath(file_obj)
        if not s.startswith(prefix):
            s = os.path.abspath(s)
            if not s.startswith(prefix):
                return None
        return s[len(prefix) :]

    orig_open = builtins.open
    orig_gzip_open = gzip.open
    orig_path_open = Path.open
    orig_path_read_text = Path.read_text
    orig_path_read_bytes = Path.read_bytes

    def hooked_open(file, mode="r", *args, **kwargs):
        rel = resolve_rel(file)
        if rel is not None:
            if rel in deleted:
                raise FileNotFoundError(file)
            if rel in mem_bytes and "r" in mode and "w" not in mode and "+" not in mode:
                raw = mem_bytes[rel]
                if "b" in mode:
                    return io.BytesIO(raw)
                enc = kwargs.get("encoding", "utf-8") or "utf-8"
                nl = kwargs.get("newline")
                return io.TextIOWrapper(io.BytesIO(raw), encoding=enc, newline=nl)
        return orig_open(file, mode, *args, **kwargs)

    def hooked_gzip_open(filename, mode="rb", compresslevel=9, encoding=None, errors=None, newline=None):
        rel = resolve_rel(filename)
        if rel is not None:
            if rel in deleted:
                raise FileNotFoundError(filename)
            if rel in mem_bytes and "r" in mode and "w" not in mode and "+" not in mode:
                raw = mem_bytes[rel]
                gz = gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb")
                if "t" in mode:
                    return io.TextIOWrapper(gz, encoding=encoding or "utf-8", errors=errors or "strict", newline=newline)
                return gz
        return orig_gzip_open(filename, mode=mode, compresslevel=compresslevel, encoding=encoding, errors=errors, newline=newline)

    def hooked_path_open(self, mode="r", *args, **kwargs):
        rel = resolve_rel(self)
        if rel is not None and (rel in deleted or rel in mem_bytes):
            return hooked_open(self, mode, *args, **kwargs)
        return orig_path_open(self, mode, *args, **kwargs)

    def hooked_path_read_text(self, encoding=None, errors=None):
        rel = resolve_rel(self)
        if rel is not None:
            if rel in deleted:
                raise FileNotFoundError(self)
            if rel in mem_bytes:
                return mem_bytes[rel].decode(encoding or "utf-8", errors=errors or "strict")
        return orig_path_read_text(self, encoding=encoding, errors=errors)

    def hooked_path_read_bytes(self):
        rel = resolve_rel(self)
        if rel is not None:
            if rel in deleted:
                raise FileNotFoundError(self)
            if rel in mem_bytes:
                return mem_bytes[rel]
        return orig_path_read_bytes(self)

    builtins.open = hooked_open  # type: ignore[assignment]
    gzip.open = hooked_gzip_open  # type: ignore[assignment]
    Path.open = hooked_path_open  # type: ignore[method-assign]
    Path.read_text = hooked_path_read_text  # type: ignore[method-assign]
    Path.read_bytes = hooked_path_read_bytes  # type: ignore[method-assign]
