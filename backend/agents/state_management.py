"""
State management utilities and type definitions for the agent workflow
"""

from typing import TypedDict, Annotated, Callable, Optional


def overwrite(old: str, new: str) -> str:
    """Simple overwrite reducer for string values"""
    return new


def overwrite_list(old, new):
    """Simple overwrite reducer for list values"""
    return new


def append_unique(old: list[str], new: list[str]) -> list[str]:
    """Append unique items to a list, avoiding duplicates"""
    if not new:
        return old
    return old if old and old[-1] == new[0] else old + new


class AgentState(TypedDict, total=False):
    """State definition for the agent workflow"""

    user_query: str
    processed_query: str
    classification: Annotated[Optional[int], Callable[[str, int], int]]  # 1 or 2
    retrieved_docs: Annotated[list[str], Callable[[list[str], list[str]], list[str]]]
    final_answer: str
