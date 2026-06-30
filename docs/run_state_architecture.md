# Run State Architecture

`RunState` is the only source of truth for agent flow control. It is immutable and
can only be replaced with the result of `reduce_run_state(state, event)`.

## Phases

- `RETRIEVING`: retrieval tools only
- `ACTING`: `decision_edit` only; a validated edit may also be finalized
- `VALIDATING`: no model tools
- `RESPONDING`: final answer only
- `WAITING_USER`: no tools until the pending user decision is resolved
- `TERMINAL`: immutable; all late events are ignored

There is no stored `READY` flag. Retrieval completion is derived from the evidence
ledger:

```python
retrieval_complete = not (evidence.required - evidence.grounded)
```

## Ownership

| Component | May do | Must not do |
|---|---|---|
| Core LLM | propose tool calls or answers | mutate state or select a phase |
| Tools | return `ToolResult` data | mutate state |
| Hooks | normalize results and emit declarative event payloads | mutate state |
| Validator | return validation evidence | commit, rollback, or select a phase |
| Prompt builder | read state and artifacts | mutate either |
| Renderer | read events | mutate state |
| Reducer | produce the next `RunState` and effects | perform I/O |
| Main loop | dispatch events and execute reducer effects | directly edit RunState fields |

## Evidence And Artifacts

Evidence entries contain provenance and refer to artifact IDs. An evidence slot is
grounded only when its referenced artifact exists in `RunState.artifacts`.

Source text, schema payloads, and compact summaries live in the artifact/context
store. They are not flow state. Caches may change projection or storage format but
cannot change phase, allowed tools, completion, validation, or retry decisions.

When a file changes, the loop emits `artifacts_invalidated`. The reducer removes
the corresponding artifact references and evidence entries.

## Invariants

1. Only `_dispatch_run_event` may install a reducer-produced `RunState`.
2. `RESPONDING` executes no tool calls.
3. `VALIDATING`, `WAITING_USER`, and `TERMINAL` expose no model tools.
4. A normal answer is accepted only when `RunState.can_answer` is true.
5. Tool rejection is a failure event, not a successful tool result.
6. Step and maximum-step accounting come from `RunState` only.
7. Telemetry, event history, checklists, messages, and artifact caches cannot select
   a phase.
