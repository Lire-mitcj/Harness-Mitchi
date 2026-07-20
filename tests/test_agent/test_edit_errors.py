from __future__ import annotations

from src.agent.edit_errors import (
    EditErrorClass,
    RetryOwner,
    classify_edit_error,
    core_hint_for,
    edit_inner_retry_allowed,
    retry_owner_for,
)


def test_classify_format_and_locate() -> None:
    assert (
        classify_edit_error("invalid_patch: too many SEARCH/REPLACE blocks (4 > 3)")
        is EditErrorClass.E1_FORMAT
    )
    assert (
        classify_edit_error("mismatch: block 1 SEARCH code not found")
        is EditErrorClass.E2_LOCATE
    )
    assert (
        classify_edit_error("invalid_patch: block 2 overlaps another block")
        is EditErrorClass.E2_LOCATE
    )
    assert (
        classify_edit_error(
            "invalid_patch: decision_schema: invalid JSON: Expecting ',' delimiter"
        )
        is EditErrorClass.E1_FORMAT
    )
    assert edit_inner_retry_allowed(
        "invalid_patch: decision_schema: invalid JSON: Expecting ',' delimiter",
        edit_retries_remaining=True,
    )


def test_classify_syntax_vs_format_residue() -> None:
    assert (
        classify_edit_error(
            "AST validation failed: python_syntax_error",
            attempted_content="x = 1\n",
            apply_succeeded=True,
        )
        is EditErrorClass.E3_SYNTAX
    )
    assert (
        classify_edit_error(
            "AST validation failed: python_syntax_error",
            attempted_content=">>>>>>> REPLACE<<<<<<< SEARCH\n",
            apply_succeeded=True,
        )
        is EditErrorClass.E1_FORMAT
    )


def test_retry_owner_routing() -> None:
    assert (
        retry_owner_for(EditErrorClass.E1_FORMAT, edit_retries_remaining=True)
        is RetryOwner.EDIT
    )
    assert (
        retry_owner_for(EditErrorClass.E2_LOCATE, edit_retries_remaining=False)
        is RetryOwner.CORE
    )
    assert (
        retry_owner_for(EditErrorClass.E4_SPEC, edit_retries_remaining=True)
        is RetryOwner.CORE
    )
    assert (
        retry_owner_for(EditErrorClass.E6_MILESTONE, edit_retries_remaining=True)
        is RetryOwner.CORE
    )


def test_edit_inner_retry_allowed() -> None:
    assert edit_inner_retry_allowed(
        "mismatch: block 1 SEARCH code not found",
        edit_retries_remaining=True,
    )
    assert not edit_inner_retry_allowed(
        "mismatch: block 1 SEARCH code not found",
        edit_retries_remaining=False,
    )
    assert not edit_inner_retry_allowed(
        "Action is not edit: answer",
        edit_retries_remaining=True,
    )


def test_core_hint_mentions_error_class() -> None:
    hint = core_hint_for(EditErrorClass.E2_LOCATE)
    assert "E2_LOCATE" in hint
    assert "edit_plan" in hint
