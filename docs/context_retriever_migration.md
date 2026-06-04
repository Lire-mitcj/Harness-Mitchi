# Context Retriever Migration Plan

Goal: move low-level exploration out of Executor ReAct loops and into a
deterministic retrieval layer. The LLM should reason over gathered evidence,
not decide one tiny grep at a time.

## Target Architecture

1. Context Retriever
   - Input: `user_request`, `repo_map`, `task_template`.
   - Output: `ContextPack` with relevant files, symbols, snippets,
     constraints, confidence, and missing info.

2. Planner / Agent
   - Input: `user_request + ContextPack`.
   - Output: `PatchPlan` with files to edit, target symbols, intended changes,
     and validation plan.

3. Skill Executor
   - `code_edit(plan)`: applies scoped edits.
   - `code_search(extra_query)`: allowed only when the plan names a concrete
     missing fact.
   - `validator(plan)`: runs targeted validation or reports blockers.

4. Harness / Feedback
   - Checks edits stay inside the plan.
   - Checks acceptance criteria.
   - Requests user confirmation for low-confidence or risky execution.

## Migration Phases

### Phase 1: Data Model and Retriever Skeleton

- Add `ContextPack` and `PatchPlan` data models.
- Add `ContextRetriever` that uses repo_map to aggregate symbols, files, and
  source snippets.
- Keep current Planner and Executor paths unchanged.
- Completion: retriever unit tests pass and confidence/missing-info behavior is
  deterministic.

### Phase 2: Planner Input Upgrade

- Add `ContextPack` to Planner prompt input.
- Planner emits `PatchPlan` for change requests.
- PlanGate validates PatchPlan structure before TaskTree fallback.
- Completion: planner can produce a patch plan for known-target edit requests
  without diagnose ReAct.

### Phase 3: Skill Executor

- Add skill wrappers: `code_edit`, `code_search`, and `validator`.
- Execute `PatchPlan` through skills when confidence is high.
- Keep old SubTaskExecutor ReAct as fallback.
- Completion: high-confidence edits use one plan call plus deterministic skill
  calls, not exploratory ReAct.

### Phase 4: Harness Gates and Rollout

- Gate skill execution by confidence, changed files, and validation results.
- Add feature flag for `ContextRetriever -> PatchPlan -> Skills`.
- Log both old and new path decisions.
- Completion: fallback rate, model turns, grep count, and first-token latency are
  visible in logs.

## Rollback Strategy

Each phase keeps the current TaskTree/SubTaskExecutor route intact. If
`ContextPack.confidence` is low, `PatchPlan` is invalid, or validation fails in
an ambiguous way, the harness falls back to the existing ReAct executor with the
retrieved context attached as evidence.
