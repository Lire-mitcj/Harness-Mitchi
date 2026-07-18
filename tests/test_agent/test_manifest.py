from __future__ import annotations

from src.agent.manifest import (
    EvidenceItem,
    StepManifest,
    Sufficiency,
    backfill_targets,
    compute_sufficiency,
    execution_card,
    manifest_from_slots,
    project_manifest,
    reconcile_observations,
    retrieval_profile,
    stale_needing_refresh,
    supersede_stale_with_observations,
    wiring_gap_lines,
    wiring_gap_pending_loads,
)


def _anchor(file: str, span: tuple[int, int], code: str, symbol: str = "") -> dict:
    return {"file": file, "span": [span[0], span[1]], "code": code, "symbol": symbol}


def test_manifest_from_slots_starts_all_missing() -> None:
    manifest = manifest_from_slots(["target_implementation", "relevant_schema"])

    assert {item.id for item in manifest.required_items} == {
        "target_implementation",
        "relevant_schema",
    }
    assert all(item.status == "MISSING" for item in manifest.required_items)
    assert manifest.sufficiency == Sufficiency.INSUFFICIENT


def test_target_implementation_satisfied_by_any_verbatim_anchor() -> None:
    manifest = manifest_from_slots(["target_implementation"])
    anchors = [_anchor("list.py", (10, 20), "def handler():\n    return 1", "handler")]

    projected = project_manifest(manifest, anchors, step=1, task_mode="edit")

    item = projected.required_items[0]
    assert item.status == "SATISFIED"
    assert projected.sufficiency == Sufficiency.SUFFICIENT_FOR_EDIT


def test_schema_item_needs_ddl_not_grep_line() -> None:
    manifest = manifest_from_slots(["relevant_schema"])
    grep_only = [{"file": "db/init.sql", "span": [1, 1], "match_line": "status"}]

    projected = project_manifest(manifest, grep_only, step=1, task_mode="diagnose")
    assert projected.required_items[0].status == "MISSING"

    ddl = [_anchor("db/init.sql", (1, 3), "CREATE TABLE order_timeline (id INT);")]
    projected = project_manifest(manifest, ddl, step=2, task_mode="diagnose")
    assert projected.required_items[0].status == "SATISFIED"


def test_span_coverage_satisfies_item() -> None:
    item = EvidenceItem(
        id="x", need="span", type="span", file="a.py", span=(100, 110)
    )
    manifest = StepManifest(required_items=(item,))
    anchors = [_anchor("a.py", (90, 130), "x" * 5)]

    projected = project_manifest(manifest, anchors, step=1, task_mode="edit")
    assert projected.required_items[0].status == "SATISFIED"


def test_span_not_covered_stays_missing() -> None:
    item = EvidenceItem(id="x", need="span", type="span", file="a.py", span=(100, 110))
    manifest = StepManifest(required_items=(item,))
    anchors = [_anchor("a.py", (1, 50), "code")]

    projected = project_manifest(manifest, anchors, step=1, task_mode="edit")
    assert projected.required_items[0].status == "MISSING"


def test_stale_item_without_coverage_stays_stale() -> None:
    item = EvidenceItem(
        id="x", need="stale", type="span", file="a.py", span=(1, 5), status="STALE"
    )
    manifest = StepManifest(required_items=(item,))

    projected = project_manifest(manifest, [], step=3, task_mode="edit")
    assert projected.required_items[0].status == "STALE"
    # STALE does not block sufficiency.
    assert projected.sufficiency == Sufficiency.SUFFICIENT_FOR_EDIT


def test_compute_sufficiency_missing_is_insufficient() -> None:
    manifest = StepManifest(
        required_items=(EvidenceItem(id="a", need="a", status="MISSING"),)
    )
    assert compute_sufficiency(manifest, "edit") == Sufficiency.INSUFFICIENT


def test_compute_sufficiency_ignores_failure_items_for_gating() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(id="a", need="a", status="SATISFIED"),
            EvidenceItem(id="f", need="boom", type="test_failure", status="MISSING"),
        )
    )
    # failure items are not "core" evidence, so they do not force INSUFFICIENT
    assert compute_sufficiency(manifest, "edit") == Sufficiency.SUFFICIENT_FOR_EDIT


def test_backfill_targets_attaches_span_without_satisfying() -> None:
    manifest = manifest_from_slots(["endpoint_implementation"])
    observations = [
        {
            "file": "api.py",
            "span": [5, 25],
            "symbol": "list_endpoint",
            "code": "@router.get('/x')\ndef list_endpoint(): ...",
        }
    ]

    updated = backfill_targets(manifest, observations)
    item = updated.required_items[0]
    assert item.file == "api.py"
    assert item.span == (5, 25)
    assert item.status == "MISSING"


def test_reconcile_adds_dynamic_symbol_target_from_full_tool_observation() -> None:
    manifest = manifest_from_slots(["target_implementation"])
    observation = _anchor("list.py", (10, 20), "def build_router():\n    pass", "build_router")

    reconciled = reconcile_observations(manifest, [observation])

    dynamic = [
        item for item in reconciled.required_items
        if item.id.startswith("required.symbol:")
    ]
    assert len(dynamic) == 0
    slot = next(item for item in reconciled.required_items if item.id == "target_implementation")
    assert slot.file == "list.py"
    assert slot.span == (10, 20)
    assert slot.symbol == "build_router"
    projected = project_manifest(reconciled, [observation], step=1, task_mode="edit")
    assert all(item.status == "SATISFIED" for item in projected.required_items)


def test_reconcile_adds_observed_symbol_from_bootstrap_retrieval() -> None:
    observation = _anchor("list.py", (10, 20), "def build_router():\n    pass", "build_router")

    reconciled = reconcile_observations(StepManifest(), [observation])

    matching = [
        item for item in reconciled.required_items
        if item.file == "list.py" and item.symbol == "build_router"
    ]
    assert len(matching) == 1
    assert matching[0].role == "observed"
    assert matching[0].id == "observed.symbol:list.py:build_router"


def test_reconcile_ignores_structural_grep_locator() -> None:
    manifest = manifest_from_slots(["target_implementation"])

    reconciled = reconcile_observations(
        manifest,
        [{"file": "list.py", "span": [10, 10], "match_line": "def build_router"}],
    )

    assert not any(item.id.startswith("located.") for item in reconciled.required_items)


def test_reconcile_ignores_non_structural_grep_locator() -> None:
    manifest = StepManifest()

    reconciled = reconcile_observations(
        manifest,
        [{"file": "list.py", "span": [10, 10], "match_line": "status = 'paid'"}],
    )

    assert reconciled.required_items == ()
    assert compute_sufficiency(reconciled, "edit") == Sufficiency.INSUFFICIENT


def test_execution_card_lists_missing_and_tools() -> None:
    manifest = manifest_from_slots(["relevant_schema"])
    manifest = manifest.__class__(
        required_items=(
            EvidenceItem(
                id="relevant_schema",
                need="ticket_order schema",
                type="schema",
                file="db/init.sql",
                span=(99, 110),
                status="MISSING",
            ),
        ),
        sufficiency=Sufficiency.INSUFFICIENT,
    )

    card = execution_card(manifest, ["view_symbol_code", "grep_search"])
    assert "STEP EVIDENCE" in card
    assert "edit_ready: no" in card
    assert "db/init.sql:99-110" in card
    assert "tools_available: view_symbol_code, grep_search" in card


def test_execution_card_lists_loaded_targets_once() -> None:
    target = EvidenceItem(
        id="endpoint_implementation",
        need="endpoint",
        file="list.py",
        span=(16, 350),
        symbol="build_router",
        status="SATISFIED",
    )
    duplicate = EvidenceItem(
        id="discovered.symbol:list.py:build_router",
        need="dynamic endpoint",
        file="list.py",
        span=(16, 350),
        symbol="build_router",
        status="SATISFIED",
    )
    manifest = StepManifest(
        required_items=(target, duplicate),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )

    card = execution_card(manifest, ["view_symbol_code", "decision_edit"])

    assert "loaded (reuse; do not re-fetch):" in card
    assert "edit_ready: yes" in card
    assert card.count("list.py:16-350") == 1
    assert "symbol=build_router" in card


def test_empty_manifest_card_reports_bootstrap_state() -> None:
    card = execution_card(StepManifest(), ["grep_search"])

    assert "bootstrap: no verified target loaded yet" in card
    assert ">>> NEXT:" not in card


def test_retrieval_profile_bootstrap_missing_prefers_grep_and_heavy() -> None:
    manifest = manifest_from_slots(["target_implementation"])
    profile = retrieval_profile(manifest)

    assert profile.bootstrap is True
    assert profile.needs_grep is True
    assert profile.needs_heavy is True
    assert profile.needs_view is False


def test_retrieval_profile_concrete_missing_opens_view() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="handler",
                need="handler",
                file="api.py",
                symbol="handler",
                status="MISSING",
            ),
        ),
    )
    profile = retrieval_profile(manifest)

    assert profile.needs_view is True
    assert profile.needs_grep is False


def test_retrieval_profile_stale_only_opens_view_not_grep() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="schema",
                need="schema",
                file="schema.sql",
                status="STALE",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    profile = retrieval_profile(manifest)

    assert profile.needs_view is True
    assert profile.needs_grep is False
    assert profile.needs_heavy is False


def test_stale_needing_refresh_ignores_superseded_symbol() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_order",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(99, 110),
                symbol="ticket_order",
                status="STALE",
                stale_reason="file modified by decision_edit",
            ),
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_order:fresh",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(99, 112),
                symbol="ticket_order",
                status="SATISFIED",
            ),
        ),
    )

    assert stale_needing_refresh(manifest) == ()
    profile = retrieval_profile(manifest)
    assert profile.needs_view is False


def test_supersede_stale_with_observations_promotes_matching_rows() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_order",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(99, 110),
                symbol="ticket_order",
                status="STALE",
                stale_reason="file modified by decision_edit",
            ),
            EvidenceItem(
                id="observed.schema:db/init/init.sql:airport_info",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(7, 14),
                symbol="airport_info",
                status="STALE",
                stale_reason="file modified by decision_edit",
            ),
        ),
    )
    observations = [
        _anchor(
            "db/init/init.sql",
            (99, 112),
            "CREATE TABLE ticket_order (\n  id INT PRIMARY KEY\n);",
            "ticket_order",
        ),
    ]

    updated = supersede_stale_with_observations(manifest, observations)
    ticket = next(item for item in updated.required_items if item.symbol == "ticket_order")
    airport = next(item for item in updated.required_items if item.symbol == "airport_info")

    assert ticket.status == "SATISFIED"
    assert ticket.span == (99, 112)
    assert ticket.stale_reason is None
    assert airport.status == "STALE"


def test_reconcile_supersedes_stale_schema_from_post_edit_observations() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_order",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(99, 110),
                symbol="ticket_order",
                status="STALE",
                stale_reason="file modified by decision_edit",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    observation = _anchor(
        "db/init/init.sql",
        (99, 115),
        "CREATE TABLE ticket_order (\n  id INT PRIMARY KEY,\n  status TEXT\n);",
        "ticket_order",
    )

    reconciled = reconcile_observations(manifest, [observation])
    item = next(item for item in reconciled.required_items if item.symbol == "ticket_order")

    assert item.status == "SATISFIED"
    assert item.span == (99, 115)
    assert stale_needing_refresh(reconciled) == ()


def test_execution_card_shows_stale_anchors_during_edit_burst() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_order",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(99, 110),
                symbol="ticket_order",
                status="STALE",
                stale_reason="file modified by decision_edit",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="handler",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 350),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )

    card = execution_card(
        manifest,
        ["decision_edit"],
        edit_burst=True,
        edited_files=("db/init/init.sql",),
    )

    assert "edit_burst:" in card
    assert "batch verification" in card
    assert "stale anchors" in card
    assert "db/init/init.sql:99-110" in card


def test_grep_pending_caller_loads_when_mount_file_not_grounded() -> None:
    from src.agent.manifest import grep_pending_caller_loads

    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
    )
    pending = grep_pending_caller_loads(
        manifest,
        (
            {
                "file": "main.py",
                "symbol": "wire_routes",
                "span": [3, 5],
                "resolved_from": "mount_context",
            },
        ),
    )
    assert len(pending) == 1
    assert pending[0]["file"] == "main.py"
    assert pending[0]["symbol"] == "wire_routes"


def test_observed_bootstrap_requires_schema_and_symbol_for_edit() -> None:
    schema = EvidenceItem(
        id="observed.schema:db/init/init.sql:ticket_order",
        need="schema",
        type="schema",
        role="observed",
        file="db/init/init.sql",
        span=(99, 110),
        symbol="ticket_order",
        status="SATISFIED",
    )
    symbol = EvidenceItem(
        id="observed.symbol:list.py:order_timeline",
        need="handler",
        type="symbol",
        role="observed",
        file="list.py",
        span=(346, 400),
        symbol="order_timeline",
        status="SATISFIED",
    )
    manifest = StepManifest(required_items=(schema, symbol))
    assert compute_sufficiency(manifest, "edit") == Sufficiency.SUFFICIENT_FOR_EDIT

    only_symbol = StepManifest(required_items=(symbol,))
    assert compute_sufficiency(only_symbol, "edit") == Sufficiency.SUFFICIENT_FOR_EDIT

    small_symbol = EvidenceItem(
        id="observed.symbol:list.py:helper",
        need="helper",
        type="symbol",
        role="observed",
        file="list.py",
        span=(1, 2),
        symbol="helper",
        status="SATISFIED",
    )
    assert compute_sufficiency(StepManifest(required_items=(small_symbol,)), "edit") == Sufficiency.INSUFFICIENT


def test_observed_integration_bootstrap_main_handlers_plus_build_router() -> None:
    """Large edit-target symbol is edit-ready even if caller wiring is incomplete."""
    handlers = [
        EvidenceItem(
            id=f"observed.symbol:main.py:{name}",
            need=name,
            type="symbol",
            role="observed",
            file="main.py",
            span=(49 + index * 8, 55 + index * 8),
            symbol=name,
            status="SATISFIED",
        )
        for index, name in enumerate(
            ("sqlalchemy_error_handler", "validation_error_handler", "general_error_handler")
        )
    ]
    router = EvidenceItem(
        id="observed.symbol:list.py:build_router",
        need="router",
        type="symbol",
        role="observed",
        file="list.py",
        span=(16, 358),
        symbol="build_router",
        status="SATISFIED",
    )
    manifest = StepManifest(required_items=(*handlers, router))
    assert compute_sufficiency(manifest, "edit") == Sufficiency.SUFFICIENT_FOR_EDIT
    assert wiring_gap_lines(manifest)


def test_wiring_gap_persists_when_only_single_line_app_loaded() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:app",
                need="app",
                type="symbol",
                role="observed",
                file="main.py",
                span=(41, 41),
                symbol="app",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    assert wiring_gap_lines(manifest)


def test_wiring_gap_cleared_when_wide_header_span_loaded() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:app",
                need="app",
                type="symbol",
                role="observed",
                file="main.py",
                span=(1, 45),
                symbol="app",
                status="SATISFIED",
                keywords=("mount_confirmed",),
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    assert not wiring_gap_lines(manifest)


def test_observed_integration_bootstrap_ready_with_caller_setup() -> None:
    handlers = [
        EvidenceItem(
            id="observed.symbol:main.py:create_app",
            need="app",
            type="symbol",
            role="observed",
            file="main.py",
            span=(1, 35),
            symbol="create_app",
            status="SATISFIED",
            keywords=("mount_confirmed",),
        ),
        EvidenceItem(
            id="observed.symbol:main.py:sqlalchemy_error_handler",
            need="handler",
            type="symbol",
            role="observed",
            file="main.py",
            span=(49, 55),
            symbol="sqlalchemy_error_handler",
            status="SATISFIED",
        ),
    ]
    router = EvidenceItem(
        id="observed.symbol:list.py:build_router",
        need="router",
        type="symbol",
        role="observed",
        file="list.py",
        span=(16, 358),
        symbol="build_router",
        status="SATISFIED",
    )
    manifest = StepManifest(required_items=(*handlers, router))
    assert compute_sufficiency(manifest, "edit") == Sufficiency.SUFFICIENT_FOR_EDIT
    assert not wiring_gap_lines(manifest)


def test_execution_card_reports_wiring_gap() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:sqlalchemy_error_handler",
                need="handler",
                type="symbol",
                role="observed",
                file="main.py",
                span=(49, 55),
                symbol="sqlalchemy_error_handler",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    card = execution_card(
        manifest,
        ["view_symbol_code", "decision_edit"],
        task_text="mount list.py router in main.py include_router",
    )
    assert "wiring_gap:" in card
    assert "main.py" in card
    assert "pending_wiring:" in card
    assert "edit_target: list.py" in card


def test_execution_card_filters_cross_file_duplicate_suggested_views() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    card = execution_card(
        manifest,
        ["view_symbol_code"],
        grep_suggested_views=(
            {"file": "main.py", "symbol": "build_router", "span": [42, 42]},
            {"file": "main.py", "symbol": "create_app", "span": [10, 10]},
        ),
    )
    assert "build_router" not in card.split("suggested_views", 1)[-1]
    assert "create_app" in card


def test_wiring_gap_pending_loads_suggests_caller_setup() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:sqlalchemy_error_handler",
                need="handler",
                type="symbol",
                role="observed",
                file="main.py",
                span=(49, 55),
                symbol="sqlalchemy_error_handler",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    loads = wiring_gap_pending_loads(
        manifest,
        task_text="mount list.py router in main.py include_router",
    )
    assert len(loads) == 1
    assert loads[0]["file"] == "main.py"
    assert loads[0]["symbol"] == "create_app"


def test_retrieval_profile_needs_view_on_wiring_gap() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:sqlalchemy_error_handler",
                need="handler",
                type="symbol",
                role="observed",
                file="main.py",
                span=(49, 55),
                symbol="sqlalchemy_error_handler",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    profile = retrieval_profile(manifest)
    assert profile.needs_view is True


def test_stale_observed_schema_still_allows_edit() -> None:
    schema = EvidenceItem(
        id="observed.schema:db/init/init.sql:ticket_order",
        need="schema",
        type="schema",
        role="observed",
        file="db/init/init.sql",
        span=(99, 110),
        symbol="ticket_order",
        status="STALE",
        stale_reason="file modified by decision_edit",
    )
    symbol = EvidenceItem(
        id="observed.symbol:list.py:build_router",
        need="handler",
        type="symbol",
        role="observed",
        file="list.py",
        span=(16, 350),
        symbol="build_router",
        status="SATISFIED",
    )
    manifest = StepManifest(required_items=(schema, symbol))
    assert compute_sufficiency(manifest, "edit") == Sufficiency.SUFFICIENT_FOR_EDIT


def test_execution_card_includes_discovery_hints_from_task_slots() -> None:
    card = execution_card(
        StepManifest(),
        ["grep_search", "view_symbol_code"],
        task_slots=["endpoint_implementation", "relevant_schema"],
        task_text="Add order timeline endpoint and ticket_order schema",
    )

    assert "discovery_hints" in card
    assert "endpoint_implementation" in card or "task_batch" in card
    assert "relevant_schema" in card or "CREATE TABLE" in card
    assert "not manifest obligations" in card


def test_execution_card_edit_only_includes_retrieval_closed_line() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:sqlalchemy_error_handler",
                need="handler",
                type="symbol",
                role="observed",
                file="main.py",
                span=(49, 55),
                symbol="sqlalchemy_error_handler",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="handler",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 350),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    card = execution_card(
        manifest,
        ["decision_edit"],
        task_text="wire db logging into list.py routes",
    )

    assert "retrieval_closed:" in card
    assert "edit_target: list.py" in card
    assert "LOADED CODE ANCHORS" in card
    assert "do not decision_edit other files to read" in card


def test_execution_card_surfaces_grep_error_and_suggested_views() -> None:
    card = execution_card(
        StepManifest(),
        ["grep_search", "view_symbol_code"],
        last_grep_error="grep_search requires a non-empty 'pattern' or 'patterns' list.",
        grep_suggested_views=(
            {"file": "db/init/init.sql", "symbol": "ticket_order", "span": [99, 99]},
        ),
    )
    assert "last_grep_error:" in card
    assert "suggested_views" in card
    assert "ticket_order" in card
    assert "view_symbol_code" in card


def test_wiring_gap_when_create_app_loaded_without_mount_content() -> None:
    """Symbol name create_app alone must not clear caller wiring gap."""
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:create_app",
                need="create_app",
                type="symbol",
                role="observed",
                file="main.py",
                span=(1, 80),
                symbol="create_app",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    assert wiring_gap_lines(manifest)
    profile = retrieval_profile(manifest)
    assert profile.needs_view


def test_reconcile_observations_tags_mount_confirmed_on_setup_code() -> None:
    manifest = reconcile_observations(
        StepManifest(),
        observations=[
            _anchor(
                "main.py",
                (1, 40),
                "def create_app():\n    app = FastAPI()\n    app.include_router(build_router(engine))\n",
                symbol="create_app",
            )
        ],
    )
    item = next(item for item in manifest.observed_items if item.file == "main.py")
    assert "mount_confirmed" in item.keywords


def test_cross_file_partner_files_excludes_edited_target() -> None:
    from src.agent.manifest import cross_file_partner_files

    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:create_app",
                need="app",
                type="symbol",
                role="observed",
                file="main.py",
                span=(1, 40),
                symbol="create_app",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 100),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    assert cross_file_partner_files(manifest, "list.py") == frozenset({"main.py"})


def test_wiring_gap_skipped_for_mention_yaml_tasks() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:sqlalchemy_error_handler",
                need="handler",
                type="symbol",
                role="observed",
                file="main.py",
                span=(49, 55),
                symbol="sqlalchemy_error_handler",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        )
    )
    task = "在 noise_policy.yaml 添加 bot_nicknames，并优化 @mention 检测"
    assert wiring_gap_lines(manifest)
    assert not wiring_gap_lines(manifest, task_text=task)
    profile = retrieval_profile(manifest, task_text=task)
    assert not profile.needs_view


def test_wiring_gap_skipped_between_infrastructure_peers() -> None:
    """Infra/config peers should not trigger FastAPI-style caller wiring gaps."""
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:agentmesh_orchestrator/conf/noise_policy.py:load_policy",
                need="policy",
                type="symbol",
                role="observed",
                file="agentmesh_orchestrator/conf/noise_policy.py",
                span=(1, 120),
                symbol="load_policy",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:agentmesh_orchestrator/internal/mention_rules.py:rules",
                need="rules",
                type="symbol",
                role="observed",
                file="agentmesh_orchestrator/internal/mention_rules.py",
                span=(1, 80),
                symbol="rules",
                status="SATISFIED",
            ),
        )
    )
    assert not wiring_gap_lines(manifest)
