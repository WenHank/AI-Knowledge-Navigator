from typing import Literal
import time

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

from agents.preprocessing import PreprocessingAgent
from agents.llm_openrouter import OpenrouterAgent
from agents.llm_localmodel import LocalAgent
from agents.state_management import AgentState

# Initialize agents
pre_agent = PreprocessingAgent()
ope_agent = OpenrouterAgent()
loc_agent = LocalAgent()


def preprocessing_node(
    state: AgentState,
) -> Command[Literal["simple_processing", "complex_processing"]]:
    """
    Preprocessing node that processes the user query and routes based on classification
    """

    print("------------- Starting processing node -------------")

    execution_summary = state.get("execution_summary", {})

    try:
        new_state = pre_agent(state)

        execution_summary["preprocessing"] = {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        routing_type = new_state.get("routing_type", "1")
        next_node = "simple_processing" if routing_type == "1" else "complex_processing"

        update = {
            "current_node": "preprocessing",
            "node_status": {"preprocessing": "completed"},
            "execution_summary": execution_summary,
            "routing_type": routing_type
        }

        if "preprocess_error" in new_state:
            update["preprocess_error"] = new_state["preprocess_error"]
            execution_summary["preprocessing"]["status"] = "warning"
            execution_summary["preprocessing"]["error"] = new_state["preprocess_error"]

        return Command(goto=next_node, update=update)

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
            "routing_type": "1"  # Default to simple processing on error
        }

        return Command(goto="simple_processing", update=update)


def simple_processing_node(state: AgentState) -> Command[Literal[END]]:
    """
    Simple processing node using LocalAgent for simple queries (classification = 1)
    """
    print("------------- Starting simple processing node -------------")

    execution_summary = state.get("execution_summary", {})
    user_query = state.get("user_query", "")

    try:
        # Use LocalAgent for simple processing as intended
        new_state = loc_agent(state)

        execution_summary["localagent"] = {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        final_answer = new_state.get("final_answer", "")
        
        update = {
            "current_node": "localagent",
            "node_status": {"localagent": "completed"},
            "execution_summary": execution_summary,
            "final_answer": final_answer
        }

        if "localagent_error" in new_state:
            update["localagent_error"] = new_state["localagent_error"]
            execution_summary["localagent"]["status"] = "warning"
            execution_summary["localagent"]["error"] = new_state["localagent_error"]

        return Command(goto=END, update=update)

    except Exception as e:
        execution_summary["localagent"] = {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        update = {
            "localagent_error": str(e),
            "current_node": "localagent",
            "node_status": {"localagent": "failed"},
            "execution_summary": execution_summary,
            "final_answer": f"Error occurred during simple processing of '{user_query}': {str(e)}"
        }

        return Command(goto=END, update=update)


def complex_processing_node(state: AgentState) -> Command[Literal[END]]:
    """
    Complex processing node using OpenrouterAgent for advanced queries (classification = 2)
    """
    print("------------- Starting complex processing node -------------")
    execution_summary = state.get("execution_summary", {})
    user_query = state.get("user_query", "")

    try:
        new_state = ope_agent(state)

        execution_summary["openrouteragent"] = {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        final_answer = new_state.get("final_answer", "")
        
        update = {
            "current_node": "openrouteragent",
            "node_status": {"openrouteragent": "completed"},
            "execution_summary": execution_summary,
            "final_answer": final_answer
        }

        if "openrouteragent_error" in new_state:
            update["openrouteragent_error"] = new_state["openrouteragent_error"]
            execution_summary["openrouteragent"]["status"] = "warning"
            execution_summary["openrouteragent"]["error"] = new_state["openrouteragent_error"]

        return Command(goto=END, update=update)

    except Exception as e:
        execution_summary["openrouteragent"] = {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        update = {
            "openrouteragent_error": str(e),
            "current_node": "openrouteragent",
            "node_status": {"openrouteragent": "failed"},
            "execution_summary": execution_summary,
            "final_answer": f"Error occurred during complex processing of '{user_query}': {str(e)}"
        }

        return Command(goto=END, update=update)


workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("preprocessing", preprocessing_node)
workflow.add_node("simple_processing", simple_processing_node)
workflow.add_node("complex_processing", complex_processing_node)

# Add edges
workflow.add_edge(START, "preprocessing")

workflow.set_entry_point("preprocessing")

agent_runner = workflow.compile()


# Example usage
if __name__ == "__main__":

    test_queries = [
            "What is machine learning?",  # Should be type 1 (EASY)
            # "What's the capital of France?",  # Should be type 1 (EASY) 
            "How do I make coffee?",  # Should be type 1 (EASY)
            # "Explain quantum entanglement's implications for cryptography",  # Should be type 2 (DIFFICULT)
            # "Design a distributed system architecture for handling 1M concurrent users",  # Should be type 2 (DIFFICULT)
            "Analyze the geopolitical implications of climate change on global trade patterns",  # Should be type 2 (DIFFICULT)
        ]

    for query in test_queries:
            print(f"\n--- Testing: {query[:50]}... ---")
              # Test with a simple query
            initial_state = {
                "user_query":query,
                "execution_summary": {},
                "node_status": {}
            }
            final_state = agent_runner.invoke(initial_state)

            print("Final result:", final_state)

  
    