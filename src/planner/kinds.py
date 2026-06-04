from __future__ import annotations

from enum import StrEnum


class SubTaskKind(StrEnum):
    DIAGNOSE = "diagnose"
    EDIT = "edit"
    VERIFY = "verify"
    SHELL = "shell"


def default_needs_l1(kind: SubTaskKind) -> bool:
    return kind == SubTaskKind.EDIT
