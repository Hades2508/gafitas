"""A marker inside an argument used to replace the call that contained it.

F-174, from an independent correctness review of the parsing path.

THE HIJACK
----------
`parse_text_call` took the LAST marker unconditionally. The reason is sound and
was documented -- models reason first and act last -- but nothing checked that
the chosen marker's call accounted for the rest of the message. So this:

    LLAMADA: write_file(path="n.txt", content="LLAMADA: read_file(path='o')")

which is ONE valid write, executed `read_file(path='o')` instead.

That is not a rejected call. A DIFFERENT call ran, the agent's write vanished,
and no error told it so -- the turn came back looking successful. It fires
whenever written content mentions the marker at all: documentation about this
harness, a test fixture, a Spanish comment.

A marker now qualifies when its call closes at the END of the message. The
inner one above leaves a trailing `")`, so it loses to the outer one. When
none qualifies -- ordinary prose after a real call -- the last marker is used
exactly as before, so nothing that parsed yesterday stops parsing.

THE REJECTION
-------------
    LLAMADA: read_file(path = "a.py")

was refused with `no pude leer el valor '= "a.py"'`. `_split_top_level`
returns STRIPPED pieces and the value offset was computed from their length
against the UNSTRIPPED text, so every space before the `=` moved the cut left
onto the separator itself. Spaces around `=` are ordinary, and a model writing
them lost a turn to a parser bug.
"""

from __future__ import annotations

import pytest

from localprog import protocol
from localprog.errors import InvalidCall

Q = chr(39)


def test_a_marker_inside_an_argument_does_not_take_over():
    call = protocol.parse_text_call(
        f'LLAMADA: write_file(path="notas.txt", '
        f'content="LLAMADA: read_file(path={Q}otro.txt{Q})")')
    assert call.name == "write_file", (
        "the agent asked to write a file; a different call used to run instead")
    assert call.arguments["path"] == "notas.txt"
    assert "read_file" in call.arguments["content"], (
        "the content must arrive intact -- it is what the agent meant to write")


def test_content_mentioning_only_the_marker_is_not_a_rejection():
    call = protocol.parse_text_call(
        'LLAMADA: write_file(path="notas.txt", content="pon LLAMADA: aqui")')
    assert call.name == "write_file"
    assert call.arguments["content"] == "pon LLAMADA: aqui"


def test_reasoning_before_the_call_still_works():
    """The behaviour the last-marker rule exists for. It must survive."""
    call = protocol.parse_text_call(
        'Primero voy a mirar el fichero. LLAMADA: grep(pattern="x")')
    assert call.name == "grep"


def test_prose_after_the_call_still_works():
    """No marker closes at the end here, so the fallback has to hold."""
    call = protocol.parse_text_call(
        'LLAMADA: read_file(path="a.py") y luego decidimos')
    assert call.name == "read_file"
    assert call.arguments["path"] == "a.py"


def test_two_real_calls_still_take_the_last():
    call = protocol.parse_text_call(
        'LLAMADA: read_file(path="a.py")\nLLAMADA: read_file(path="b.py")')
    assert call.arguments["path"] == "b.py"


@pytest.mark.parametrize("text,expected", [
    ('LLAMADA: read_file(path="a.py")', "a.py"),
    ('LLAMADA: read_file(path ="a.py")', "a.py"),
    ('LLAMADA: read_file(path= "a.py")', "a.py"),
    ('LLAMADA: read_file(path = "a.py")', "a.py"),
    ('LLAMADA: read_file(  path   =   "a.py"  )', "a.py"),
])
def test_spaces_around_the_equals_are_accepted(text, expected):
    assert protocol.parse_text_call(text).arguments["path"] == expected


def test_a_message_with_no_marker_is_still_refused():
    with pytest.raises(InvalidCall):
        protocol.parse_text_call("solo texto, sin llamada ninguna")


def test_an_unclosed_call_is_still_refused():
    with pytest.raises(InvalidCall):
        protocol.parse_text_call('LLAMADA: read_file(path="a.py"')
