"""Write-scope matching. A small pattern language, not a list of exact paths.

Why this module exists (audit finding F-05)
-------------------------------------------
``_check_writable`` used to be::

    allowed = set(write_scope) | set(allowed_new_files)
    if rel not in allowed: refuse

so the agent could only ever touch a path the mission author had typed out in
advance. Three consequences, all of them fatal to the goal:

  * a task that genuinely needs a new module cannot be done unless the author
    already knew the module's name -- which means the author already did the
    design work;
  * a task that needs a companion test file cannot be done at all;
  * multi-file work degenerates into "edit exactly these N files", which is a
    patch application, not programming.

The fix is to widen the *scope language*, never the containment. Containment
still happens in ``programmer.guard`` on every single path resolution: ``..``,
absolute paths, symlinks and device names are rejected there, before anything
in this module is consulted. This module only answers a narrower question,
about a path that is already known to be inside the repository:

    is this relative path one the mission said the agent may write?

Three entry forms, chosen because they are the three things mission authors
actually mean:

    "pkg/mod.py"    exactly that file
    "pkg/"          that directory and everything beneath it
    "pkg/*.py"      a glob, matched with fnmatch semantics where ``*`` does
                    not cross a directory separator and ``**/`` does

Everything is compared in posix form and case-sensitively. Windows is
case-insensitive on disk, but a scope that matched ``PKG/Mod.py`` for
``pkg/mod.py`` would be a scope that behaves differently on two machines, and
a containment rule that is platform-dependent is not a containment rule.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass


def _posix(text: str) -> str:
    return text.replace("\\", "/").strip()


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """fnmatch, except ``*`` stops at ``/`` and ``**`` does not.

    ``fnmatch.translate`` maps ``*`` to ``.*``, which makes ``src/*.py`` match
    ``src/a/b/c.py``. For a write scope that is the difference between "this
    package" and "this package and every package under it", so it is worth the
    twenty lines.
    """
    out = ["(?s:"]
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")      # zero or more directories
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if c == "?":
            out.append("[^/]")
            i += 1
            continue
        if c == "[":
            close = pattern.find("]", i + 1)
            if close == -1:
                out.append(re.escape(c))
                i += 1
                continue
            body = pattern[i + 1 : close]
            body = body.replace("\\", "\\\\")
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = close + 1
            continue
        out.append(re.escape(c))
        i += 1
    out.append(r")\Z")
    return re.compile("".join(out))


@dataclass(frozen=True)
class Rule:
    r"""One scope entry, remembering the text the mission actually wrote.

    ``source`` is kept because the refusal message shows the model the scope it
    was given, verbatim. A model that is told ``'src/'`` can reason about it; a
    model told ``re.compile('(?s:src/.*)\\Z')`` cannot.
    """

    source: str
    kind: str            # FILE | DIR | GLOB
    _regex: re.Pattern[str] | None = None
    _prefix: str = ""

    def matches(self, rel: str) -> bool:
        if self.kind == "FILE":
            return rel == self.source
        if self.kind == "DIR":
            return rel.startswith(self._prefix)
        return self._regex is not None and self._regex.match(rel) is not None


def compile_rule(entry: str) -> Rule:
    text = _posix(entry)
    if not text:
        raise ValueError("empty scope entry")
    if text.endswith("/"):
        return Rule(source=text, kind="DIR", _prefix=text)
    if any(ch in text for ch in "*?["):
        return Rule(source=text, kind="GLOB", _regex=_glob_to_regex(text))
    return Rule(source=text, kind="FILE")


class WriteScope:
    """The set of relative paths a mission permits writing to.

    ``allowed_new_files`` is kept separate from ``write_scope`` for one reason:
    a mission may want to say "you may edit these files, and additionally you
    may create these". Creation consults both; modification consults only the
    first. That distinction was in the original design and is worth keeping --
    it is how a mission expresses "do not touch the tests, but you may add a
    new one".
    """

    def __init__(
        self,
        write_scope: tuple[str, ...] = (),
        allowed_new_files: tuple[str, ...] = (),
    ) -> None:
        self.write_entries = tuple(_posix(e) for e in write_scope if _posix(e))
        self.new_entries = tuple(_posix(e) for e in allowed_new_files if _posix(e))
        self._write = [compile_rule(e) for e in self.write_entries]
        self._new = [compile_rule(e) for e in self.new_entries]

    def __bool__(self) -> bool:
        return bool(self._write or self._new)

    def allows(self, rel: str, *, creating: bool) -> bool:
        rel = _posix(rel)
        rules = self._write + self._new if creating else self._write
        return any(rule.matches(rel) for rule in rules)

    def describe(self, *, creating: bool) -> str:
        """What to show the model when it is refused. Its own words, back."""
        entries = list(self.write_entries)
        if creating:
            entries += [e for e in self.new_entries if e not in entries]
        return ", ".join(entries) or "(nada)"

    def to_list(self) -> list[str]:
        return list(dict.fromkeys(self.write_entries + self.new_entries))
