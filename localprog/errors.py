"""The four-class error taxonomy. This module is the whole point of the harness.

Every failure that can happen while driving a model belongs to exactly one of
four classes, and each class has exactly one destiny:

    ToolError      -> structured feedback text handed back to the model.
                      The loop CONTINUES. Never counted as an invalid call.

    InvalidCall    -> the MODEL emitted something unusable (no call, unknown
                      tool, arguments that are not an object, a required
                      argument missing). Counted in ``llamadas_invalidas`` per
                      the frozen SCREEN_V0 scoring, AND fed back so the model
                      can recover. The loop CONTINUES.

    ProviderError  -> the model server failed (HTTP status, timeout, body that
                      is not JSON, no message). The run ends with a structured,
                      persisted PROVIDER_ERROR result. NEVER scored as a model
                      failure, because it is not one.

    HarnessInvalid -> a real defect in THIS code. The only thing in the whole
                      system that may produce HARNESS_INVALID. The run is
                      voided, the traceback is preserved, and the workspace is
                      kept for inspection.

The rule that makes the taxonomy honest: ``tools.dispatch`` converts ANY
exception that is not one of the first three into ``HarnessInvalid``. So a tool
that forgets to raise ``ToolError`` for a foreseeable condition does not quietly
corrupt a score -- it stops the run and names itself. That is why the tools in
``tools.py`` are written as total functions.
"""

from __future__ import annotations


class LocalProgError(Exception):
    """Base for everything this harness raises deliberately."""


class ToolError(LocalProgError):
    """An expected condition inside a tool.

    This is not a bug and not a model mistake -- it is the tool reporting a
    fact about the repository ("that file does not exist", "that text appears
    three times"). The message IS the contract: it is what the model reads and
    reacts to, and LOCAL_PROGRAMMER_V0_CONTRACT.md specifies the text of the
    important ones. Deterministic feedback is the mechanism the whole design
    bets on, so these strings are production behaviour, not diagnostics.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)

    def feedback(self) -> str:
        return f"{self.code}: {self.detail}" if self.detail else self.code


class InvalidCall(LocalProgError):
    """The model did not produce a usable tool call.

    Distinct from ToolError on purpose. The frozen scoring counts these
    (``llamadas_invalidas`` = "herramienta inexistente o argumentos mal
    formados") and does not count ToolErrors. Conflating them would silently
    change a frozen metric, so they are different types and are counted in
    different places.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)

    def feedback(self) -> str:
        return f"{self.code}: {self.detail}" if self.detail else self.code


class ProviderError(LocalProgError):
    """The model server failed to answer.

    ``kind`` is a closed set so the persisted result can be grouped without
    parsing prose:

        HTTP_STATUS   the server answered with a non-2xx status
        TIMEOUT       the call exceeded the deadline
        BAD_BODY      the body was not JSON, or had no message
        TRANSPORT     the connection itself failed (refused, reset, DNS)
    """

    KINDS = ("HTTP_STATUS", "TIMEOUT", "BAD_BODY", "TRANSPORT")

    def __init__(self, kind: str, detail: str, *, status: int | None = None) -> None:
        if kind not in self.KINDS:
            # A wrong kind here is a defect in this harness, not a provider
            # fault, and it must not be able to disguise itself as one.
            raise HarnessInvalid(f"unknown ProviderError kind {kind!r}")
        self.kind = kind
        self.detail = detail
        self.status = status
        super().__init__(f"PROVIDER_{kind}: {detail}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "status": self.status}


class HarnessInvalid(LocalProgError):
    """A defect in this harness. Voids the run; never scores anything.

    Deliberately the narrowest class. If this is raised, the correct response
    is to fix this repository -- not to retry, not to blame the model, and not
    to record a result.
    """

    def __init__(self, detail: str, *, traceback_text: str | None = None) -> None:
        self.detail = detail
        self.traceback_text = traceback_text
        super().__init__(f"HARNESS_INVALID: {detail}")

    def to_dict(self) -> dict:
        return {"detail": self.detail, "traceback": self.traceback_text}


# ---------------------------------------------------------------- tool codes
#
# The closed set of ToolError codes. Listed here rather than spelled inline at
# each raise site so that the test suite can assert the set is exactly this --
# a new code appearing without a test is a contract drift, and the suite says
# so instead of discovering it during a measurement.

ERROR_PATH_OUTSIDE_REPO = "ERROR_PATH_OUTSIDE_REPO"
ERROR_FILE_NOT_FOUND = "ERROR_FILE_NOT_FOUND"
ERROR_IS_DIRECTORY = "ERROR_IS_DIRECTORY"
ERROR_NOT_TEXT = "ERROR_NOT_TEXT"
ERROR_BAD_PATTERN = "ERROR_BAD_PATTERN"
ERROR_SYNTAX = "ERROR_SYNTAX"
ERROR_EMPTY_OLD = "ERROR_EMPTY_OLD"
ERROR_NO_MATCH = "ERROR_NO_MATCH"
ERROR_MULTIPLE_MATCHES = "ERROR_MULTIPLE_MATCHES"
ERROR_SYNTAX_AFTER_EDIT = "ERROR_SYNTAX_AFTER_EDIT"
ERROR_NOT_IN_WRITE_SCOPE = "ERROR_NOT_IN_WRITE_SCOPE"
ERROR_FILE_EXISTS = "ERROR_FILE_EXISTS"
ERROR_NOTHING_CHANGED = "ERROR_NOTHING_CHANGED"
#: ``run`` was handed an executable outside the interpreter allowlist. A
#: ToolError and not an InvalidCall: reaching for ``npm`` is a reasonable
#: thing for a model to try, and the refusal is a fact about this sandbox
#: rather than a malformed call.
ERROR_COMMAND_NOT_ALLOWED = "ERROR_COMMAND_NOT_ALLOWED"
#: ``list_dir`` on a directory that exists and holds nothing. Kept distinct
#: from ERROR_FILE_NOT_FOUND so the model does not go hunting for a typo it
#: did not make.
ERROR_EMPTY_DIRECTORY = "ERROR_EMPTY_DIRECTORY"
#: A listing was truncated. The model is told how many it did not see, so it
#: can narrow instead of assuming it saw everything.
ERROR_TOO_MANY_ENTRIES = "ERROR_TOO_MANY_ENTRIES"

TOOL_ERROR_CODES = frozenset({
    ERROR_PATH_OUTSIDE_REPO,
    ERROR_FILE_NOT_FOUND,
    ERROR_IS_DIRECTORY,
    ERROR_NOT_TEXT,
    ERROR_BAD_PATTERN,
    ERROR_SYNTAX,
    ERROR_EMPTY_OLD,
    ERROR_NO_MATCH,
    ERROR_MULTIPLE_MATCHES,
    ERROR_SYNTAX_AFTER_EDIT,
    ERROR_NOT_IN_WRITE_SCOPE,
    ERROR_FILE_EXISTS,
    ERROR_NOTHING_CHANGED,
    ERROR_COMMAND_NOT_ALLOWED,
    ERROR_EMPTY_DIRECTORY,
    ERROR_TOO_MANY_ENTRIES,
})

# ------------------------------------------------------------- invalid codes

ERROR_NO_TOOL_CALL = "ERROR_NO_TOOL_CALL"
ERROR_UNKNOWN_TOOL = "ERROR_UNKNOWN_TOOL"
ERROR_BAD_ARGUMENTS = "ERROR_BAD_ARGUMENTS"
ERROR_MISSING_ARGUMENT = "ERROR_MISSING_ARGUMENT"
ERROR_FORMAT = "ERROR_FORMAT"

INVALID_CALL_CODES = frozenset({
    ERROR_NO_TOOL_CALL,
    ERROR_UNKNOWN_TOOL,
    ERROR_BAD_ARGUMENTS,
    ERROR_MISSING_ARGUMENT,
    ERROR_FORMAT,
})
