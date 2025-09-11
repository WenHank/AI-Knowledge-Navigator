import os
import json
import time
from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

from agents.preprocessing import PreprocessingAgent
from agents.llm_openrouter import OpenrouterAgent
from agents.state_management import AgentState

pre_agent = PreprocessingAgent()
ope_agent = OpenrouterAgent()


def preprocessing_node(state: AgentState) -> AgentState:
    """Node for preprocessing user queries and determining complexity"""
    execution_summary = state.get("execution_summary", {})

    try:
        # Run preprocessing agent
        new_state = pre_agent(state)

        execution_summary["preprocessing"] = {
            "status": "success",
            "llm_type": new_state.get("preprocessing_result", "1"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        update = {
            "current_node": "preprocessing",
            "node_status": {"preprocessing": "completed"},
            "execution_summary": execution_summary,
            "preprocessing_result": new_state.get("preprocessing_result", "1"),
        }

        if "preprocess_error" in new_state:
            update["preprocess_error"] = new_state["preprocess_error"]
            execution_summary["preprocessing"]["status"] = "warning"
            execution_summary["preprocessing"]["error"] = new_state["preprocess_error"]

    except Exception as e:
        execution_summary["preprocessing"] = {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        update = {
            "preprocess_error": str(e),
            "current_node": "preprocessing",
            "node_status": {"preprocessing": "failed"},
            "execution_summary": execution_summary,
            "preprocessing_result": "1",  # Default fallback
        }

    # Determine next node based on complexity
    complexity = update.get("preprocessing_result", "1")
    if complexity == "2":
        next_node = "complex_processing"
    else:
        next_node = "simple_processing"

    return Command(update=update, goto=next_node)


def simple_processing_node(state: AgentState) -> AgentState:
    """Node for handling simple queries with lightweight processing"""
    execution_summary = state.get("execution_summary", {})

    try:
        # Use a simple/fast model or processing for easy queries
        new_state = ope_agent(state)

        execution_summary["simple_processing"] = {
            "status": "success",
            "model_used": "simple",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        update = {
            "current_node": "simple_processing",
            "node_status": {"simple_processing": "completed"},
            "execution_summary": execution_summary,
            "final_answer": new_state.get("final_answer", "No response"),
        }

    except Exception as e:
        execution_summary["simple_processing"] = {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        update = {
            "processing_error": str(e),
            "current_node": "simple_processing",
            "node_status": {"simple_processing": "failed"},
            "execution_summary": execution_summary,
        }

    return Command(update=update, goto=END)


def complex_processing_node(state: AgentState) -> AgentState:
    """Node for handling complex queries with advanced processing"""
    execution_summary = state.get("execution_summary", {})

    try:
        # Use a more powerful model or processing for complex queries
        new_state = ope_agent(state)

        execution_summary["complex_processing"] = {
            "status": "success",
            "model_used": "advanced",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        update = {
            "current_node": "complex_processing",
            "node_status": {"complex_processing": "completed"},
            "execution_summary": execution_summary,
            "final_answer": new_state.get("final_answer", "No response"),
        }

    except Exception as e:
        execution_summary["complex_processing"] = {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        update = {
            "processing_error": str(e),
            "current_node": "complex_processing",
            "node_status": {"complex_processing": "failed"},
            "execution_summary": execution_summary,
        }

    return Command(update=update, goto=END)


# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("preprocessing", preprocessing_node)
graph.add_node("simple_processing", simple_processing_node)
graph.add_node("complex_processing", complex_processing_node)

# Add edges
graph.add_edge(START, "preprocessing")
# The preprocessing node will handle routing via Command.goto

# Set entry point
graph.set_entry_point("preprocessing")

# Compile the graph
agent_runner = graph.compile()

if __name__ == "__main__":
    start_time = time.time()

    # Test with different query complexities
    test_queries = [
        "What is the capital of France?",  # Should route to simple
        "Explain quantum computing and its implications for cryptography",  # Should route to complex
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print(f"{'='*60}")

        init_state: AgentState = {
            "user_query": query,
            "execution_summary": {},
            "node_status": {},
        }

        try:
            final_state = agent_runner.invoke(init_state)

            print(
                f"\nRouting decision: {final_state.get('preprocessing_result', 'unknown')}"
            )
            print(f"Final answer: {final_state.get('final_answer', 'No answer')}")
            print(
                f"Execution summary: {json.dumps(final_state.get('execution_summary', {}), indent=2)}"
            )

        except Exception as e:
            print(f"Error running graph: {e}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
