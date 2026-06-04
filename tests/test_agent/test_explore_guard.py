from __future__ import annotations

from src.agent.explore_guard import ExploreCommandTracker, normalize_grep_query


def test_explore_tracker_blocks_duplicate_grep() -> None:
    tracker = ExploreCommandTracker(grep_dedup_limit=1)
    tracker.record_grep("boarding", path="app.py")
    deny = tracker.check_grep("boarding", path="app.py")
    assert deny is not None
    assert "duplicate grep_search" in deny


def test_explore_tracker_seeds_from_digest() -> None:
    digest = (
        "Session exploration summary\n"
        "Grep queries already run:\n"
        "  - 'boarding' in app.py\n"
        "Line ranges already read:\n"
        "  - app.py:1584-1655\n"
    )
    tracker = ExploreCommandTracker(grep_dedup_limit=1, read_dedup_limit=1)
    tracker.seed_from_digest(digest)
    assert tracker.check_grep("boarding", path="app.py") is not None
    assert tracker.check_read("app.py", start_line=1584, end_line=1655) is not None


def test_normalize_grep_query_stable() -> None:
    assert normalize_grep_query("  foo  ", path="app.py") == "grep:foo@app.py"
    assert normalize_grep_query("foo", path="./app.py") == "grep:foo@app.py"
    assert normalize_grep_query("foo", path=".", include="*.py") == "grep:foo@*.py"


def test_normalize_grep_query_canonicalizes_regex_escaping() -> None:
    a = r'@app\.get\(\"/api/orders/query\"\)'
    b = r'@app\.get\("/api/orders/query"\)'
    assert normalize_grep_query(a, path="main.py") == normalize_grep_query(b, path="main.py")


def test_duplicate_grep_dedup_path_variants() -> None:
    tracker = ExploreCommandTracker(grep_dedup_limit=1)
    tracker.record_grep("boarding", path="app.py")
    assert tracker.check_grep("boarding", path="./app.py") is not None
    assert tracker.check_grep("boarding", path=".", include="*.py") is None


def test_explore_tracker_blocks_duplicate_map_search() -> None:
    tracker = ExploreCommandTracker(map_dedup_limit=1)
    tracker.record_map("boarding_pass_view")
    deny = tracker.check_map("boarding_pass_view")
    assert deny is not None
    assert "duplicate map_search" in deny


def test_explore_tracker_seeds_map_from_digest() -> None:
    digest = (
        "Session exploration summary\n"
        "Repo map searches already run:\n"
        "  - boarding_pass_view\n"
    )
    tracker = ExploreCommandTracker(map_dedup_limit=1)
    tracker.seed_from_digest(digest)
    assert tracker.check_map("boarding_pass_view") is not None
