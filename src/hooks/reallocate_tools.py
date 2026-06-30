from __future__ import annotations
from typing import Any, TYPE_CHECKING
from src.agent.run_state import RunPhase

if TYPE_CHECKING:
    from src.agent.state_assembled_loop import AssembledState

def determine_allowed_tools(
    state: AssembledState | Any,
    gravity_controller: Any,
    default_tools: frozenset[str],
    has_compile_error: bool = False,
) -> frozenset[str]:
    """
    Hook to dynamically determine allowed tools based on retrieval completeness signals.
    
    If retrieval completeness is high (evidence saturation is high / retrieval phase is CLOSED),
    we restrict the tools. But if there is a compile/validation error, we bypass the restriction
    to allow the LLM to search/fix the issue.
    """
    if has_compile_error:
        return default_tools
        
    is_closed = getattr(gravity_controller, "retrieval_disabled", False)
    retrieval_complete = state.run_state.retrieval_complete if hasattr(state, "run_state") else False
    phase = state.run_state.phase if hasattr(state, "run_state") else None
    
    # Check if LLM has proposed a plan or if we've entered edit phase/retrieval complete
    has_plan = False
    messages = getattr(state, "messages_history", [])
    for msg in messages:
        if getattr(msg, "role", None) == "assistant" and getattr(msg, "content", None):
            content_lower = msg.content.casefold()
            # If the LLM says it has a plan, wants to implement, or is ready to edit/apply
            if any(kw in content_lower for kw in [
                "plan:", "plan\n", "here's the plan", "solution design",
                "i'll modify", "let's implement", "i will implement", "let me implement",
                "let's add", "i will add", "let me add",
                "let me apply", "let's apply", "i will apply", "i'll apply",
                "apply the fix", "apply the edit", "apply the patch",
                "apply the modification", "directly produce the edit",
                "implement this:", "modify the", "add a ", "add an "
            ]):
                has_plan = True
                break

    # If the retrieval completeness is high, or if we are in Responding or Acting phase, or if LLM has a plan
    if is_closed or retrieval_complete or phase in (RunPhase.RESPONDING, RunPhase.ACTING) or has_plan:
        task_mode = state.run_state.task_mode if hasattr(state, "run_state") else "diagnose"
        # If the assistant explicitly has a plan to edit/modify, we treat the task mode as "edit" to enable editing
        if task_mode == "edit" or has_plan:
            # For edit mode, restrict to decision_edit ONLY to force editing and avoid redundant reads
            return frozenset({"decision_edit"})
        else:
            # For diagnose mode, we allow view_symbol_code and decision_edit (to inspect and edit if needed)
            # while still enforcing search convergence (no grep_search or codebase_retrieve)
            return frozenset({"view_symbol_code", "decision_edit"})
            
    return default_tools
