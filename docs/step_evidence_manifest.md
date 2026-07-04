# Step Evidence Manifest

The Step Evidence Manifest is a system-maintained checklist of *verifiable*
evidence items for the current step. It replaces coarse evidence-slot grounding
as the driver for closing retrieval and gating tools. The Core LLM never
maintains it and never sees its decision logic; it only consumes a short,
non-prescriptive "execution card".

## What it is

Each item states one need with an objective location and a status:

- `status`: `MISSING | SATISFIED | STALE`
- `type`: `span | symbol | schema | mount | test_failure | tool_error`
- `target`: `file` (+ optional `span`, `symbol`), `content_hash`
- `role`: `required | observed`

An item is only `SATISFIED` when a durable *verbatim* code anchor covers its
target (span coverage, full symbol slice, or verbatim DDL). A grep hit alone
never satisfies an item.

Only `required` items contribute to weighted coverage and sufficiency. `observed`
items are durable retrieval memory used for deduplication. `sufficiency` is
computed from required items:
`INSUFFICIENT | SUFFICIENT_FOR_EDIT | SUFFICIENT_FOR_VERIFY | SUFFICIENT_FOR_ANSWER`.

## Functional layering

| Layer | Responsibility | Location |
| --- | --- | --- |
| Data model + pure functions | schema, status projection, sufficiency, execution card | `src/agent/manifest.py` |
| Lifecycle (reducer) | bootstrap empty, reconcile discovered targets, mark STALE, append/clear failures | `src/agent/run_state.py` |
| Projection + injection (loop) | project status from `context_anchors`, inject execution card | `src/agent/state_assembled_loop.py` |
| Tool decision | choose `allowed_tools` from `manifest.sufficiency` | `src/hooks/reallocate_tools.py` |
| Fact interception | block re-reading SATISFIED, allow STALE | `src/hooks/preflight/fact_locking.py` (via `src/hooks/before_tool.py`) |
| Parameter validation only | static argument checks, no policy | `src/hooks/preflight/static_constraints.py` |

## Who maintains it and when

Maintained by the reducer/harness, never by the Core LLM.

- Run start (`start_run`): create an empty bootstrap manifest. An empty manifest
  is explicitly `INSUFFICIENT`, so the first turn is retrieval-only without
  inventing abstract `?` targets from task keywords.
- After a successful tool (`evidence_stored`): the reducer calls the pure
  `reconcile_observations` transform. Structural grep locators (`def`, `class`,
  `CREATE TABLE/VIEW/TRIGGER/...`) become concrete `MISSING` items; ordinary
  keyword hits remain hints. Full symbol/DDL observations become discovered
  items. Reconciliation never sets `SATISFIED`.
- Before allocating tools each turn (loop): `project_manifest` recomputes each
  item status from durable code anchors and recomputes `sufficiency`.
- After an edit (`edit_applied` / `artifacts_invalidated`): items on the changed
  files are marked `STALE`.
- After failed validation (`validation_finished`): a `test_failure` item is
  appended; it is cleared on the next passing validation.

Layer ownership is strict: tools and the loop only emit observations; the
reducer is the sole lifecycle writer; `manifest.py` contains pure transforms;
tool allocation and fact locking only read the projected manifest.

## Status rules (computed)

- Span coverage (strongest): same file + a verbatim anchor whose span covers the
  item span.
- Symbol coverage: full verbatim slice of the target symbol.
- Schema: verbatim DDL (`CREATE TABLE/VIEW/TRIGGER/...`) block.
- Bootstrap emptiness is `INSUFFICIENT`; targets are learned from actual tool
  observations rather than pre-seeded semantic categories.
- STALE: set by the reducer when the target file is edited; kept STALE by the
  projection until a fresh covering anchor appears. STALE does not block
  sufficiency but lets `reallocate_tools` reopen retrieval for one refresh.

## Tool decision rules (`reallocate_tools`)

Top-down priority; the core input is `manifest.sufficiency` (not the legacy
`evidence.complete`):

- (A) concrete validation/tool error -> edit (+ retrieval only if STALE/MISSING).
- (B) `INSUFFICIENT` -> retrieval tools only; `decision_edit` withheld (no
  edit-as-read).
- (C) `SUFFICIENT_FOR_EDIT` with zero no-gain rounds -> edit plus retrieval.
- (D) After one consecutive retrieval round with no new durable observation ->
  edit plus `view_symbol_code` only.
- (E) After two consecutive no-gain rounds -> `decision_edit` only.
- (F) A new durable observation resets the no-gain counter; `MISSING`, `STALE`,
  and validation failures override convergence as appropriate.
- (G) `STALE` after an edit overrides convergence and reopens retrieval for the
  modified target.
- (H) `SUFFICIENT_FOR_ANSWER` (diagnose) -> no tools (final answer); if STALE,
  reopen retrieval instead.

## What enters the Core LLM context

Only the short execution card (`execution_card`):

```
### STEP EVIDENCE (current step)
step: task.default
edit_ready: no
required_coverage: 0.00
retrieval_no_gain_rounds: 0
bootstrap: no verified target loaded yet
tools_available: grep_search, view_symbol_code, codebase_retrieve
```

After structural discovery:

```
### STEP EVIDENCE (current step)
step: task.default
edit_ready: no
required_coverage: 0.00
retrieval_no_gain_rounds: 0
missing:
- db/init/init.sql:99-99  (结构目标待完整加载：ticket_order)
tools_available: grep_search, view_symbol_code
```

The card also includes a bounded, de-duplicated `loaded (reuse; do not
re-fetch)` list. This gives the Core LLM an explicit retrieval memory so it can
reuse grounded targets instead of proposing calls that fact locking would
reject. `STALE` remains an explicit exception that may be refreshed.

`edit_ready: yes` is a factual projection of manifest sufficiency. Together
with the repo map's file/symbol/span index, it tells the Core LLM that loaded
target and boundary anchors are enough for a scoped edit; a full-file overview
is neither required nor evidence-producing.

Excluded from context: hashes, coverage/similarity algorithms, gravity/novelty
scores, and imperative `NEXT_ACTION` / `FORBIDDEN_THIS_TURN` commands.

## Relationship to the legacy `EvidenceLedger`

`EvidenceLedger` is retained (its `evidence_stored` phase transitions still fire)
but is no longer the authority for closing retrieval or gating tools. All
tool-allocation and fact-locking decisions now read the manifest.
