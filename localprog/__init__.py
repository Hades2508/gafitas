"""LOCAL_PROGRAMMER_V0 -- the permanent harness.

One loop, seven tools, two protocols, four error classes. Built once so that
campaigns stop shipping their own runner and stop discovering harness bugs with
real models.

Implemented by Claude under an explicit operator exception (generalista.md
exception 4, with a component of exception 3: trusted implementation of safety
and infrastructure). Recorded here because the rule requires the record.

The frozen contract this implements is
``D:\\GATE-A-WS\\LOCAL_PROGRAMMER_V0\\LOCAL_PROGRAMMER_V0_CONTRACT.md``.
Behaviour that the contract fixes -- prompts, scoring, thresholds, the screen
repo -- is reproduced here and must not be edited to make a measurement come
out differently. ``INVARIANTS.md`` states what may never change.
"""

HARNESS_ID = "LOCAL_PROGRAMMER_V0"
CONTRACT = r"D:\GATE-A-WS\LOCAL_PROGRAMMER_V0\LOCAL_PROGRAMMER_V0_CONTRACT.md"

__all__ = ["HARNESS_ID", "CONTRACT"]
