"""State management utilities and type definitions for the agent workflow"""

from typing import TypedDict, Annotated, Optional

def overwrite(old: str, new: str) -> str:
    """Simple overwrite reducer for string values"""
    return new

def overwrite_int(old: Optional[int], new: Optional[int]) -> Optional[int]:
    """Simple overwrite reducer for int values"""
    return new if new is not None else old


def overwrite_list(old: list[str], new: list[str]) -> list[str]:
    """Simple overwrite reducer for list values"""
    return new if new else old


def append_unique(old: list[str], new: list[str]) -> list[str]:
    """Append unique items to a list, avoiding duplicates"""
    if not old:
        old = []
    if not new:
        return old
    # Avoid duplicates by checking if the last item in old equals the first in new
    return old if old and new and old[-1] == new[0] else old + new


class AgentState(TypedDict, total=False):
    """State definition for the agent workflow"""

    user_query: str
    final_answer: str
    routing_type: str  # "1" for simple, "2" for complex
    next_node: str

    current_node: Annotated[str, overwrite]
    node_status: Annotated[dict, overwrite]
    execution_summary: Annotated[dict, overwrite]
