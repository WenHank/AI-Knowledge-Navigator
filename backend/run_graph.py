from typing import Literal
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
    user_query = state.get("user_query", "")

    # Use the preprocessing agent to process and classify the query
    # Assuming your PreprocessingAgent has methods like process_query and classify_query
    try:
        # If your preprocessing agent has specific methods, use them here
        # For now, using the existing logic as fallback
        processed_query = user_query.strip().lower()

        # Classify the query complexity (1 = simple, 2 = complex)
        if len(processed_query) > 100 or any(
            keyword in processed_query
            for keyword in ["complex", "advanced", "detailed", "analyze", "compare"]
        ):
            classification = 2
            next_node = "complex_processing"
        else:
            classification = 1
            next_node = "simple_processing"
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Fallback to simple processing
        processed_query = user_query
        classification = 1
        next_node = "simple_processing"

    # Update state with preprocessing results
    updated_state = {
        "processed_query": processed_query,
        "classification": classification,
    }

    # Return Command to route to the appropriate processing node
    return Command(goto=next_node, update=updated_state)


def simple_processing_node(state: AgentState) -> AgentState:
    """
    Simple processing node using LocalAgent for straightforward queries (classification = 1)
    """
    processed_query = state.get("processed_query", "")

    try:
        # Use local agent for simple processing
        # Assuming your LocalAgent has a method to process queries
        # You'll need to adapt this based on your actual LocalAgent interface

        # Simple document retrieval simulation
        retrieved_docs = [
            f"Local document 1 related to: {processed_query}",
            f"Basic local information about: {processed_query}",
        ]

        # Generate simple answer using local agent
        final_answer = f"Local processing of '{processed_query}': Straightforward response with {len(retrieved_docs)} relevant documents."

    except Exception as e:
        print(f"Error in simple processing: {e}")
        retrieved_docs = [f"Error processing: {processed_query}"]
        final_answer = f"Error occurred during simple processing of '{processed_query}'"

    return {"retrieved_docs": retrieved_docs, "final_answer": final_answer}


def complex_processing_node(state: AgentState) -> AgentState:
    """
    Complex processing node using OpenrouterAgent for advanced queries (classification = 2)
    """
    processed_query = state.get("processed_query", "")

    try:
        # Use openrouter agent for complex processing
        # Assuming your OpenrouterAgent has a method to process queries
        # You'll need to adapt this based on your actual OpenrouterAgent interface

        # Complex document retrieval simulation
        retrieved_docs = [
            f"Advanced cloud document 1 with detailed analysis of: {processed_query}",
            f"Comprehensive cloud research on: {processed_query}",
            f"Technical cloud documentation for: {processed_query}",
            f"In-depth cloud study related to: {processed_query}",
        ]

        # Generate complex answer using openrouter agent
        final_answer = f"Cloud-based comprehensive analysis of '{processed_query}': Detailed response incorporating {len(retrieved_docs)} specialized documents."

    except Exception as e:
        print(f"Error in complex processing: {e}")
        retrieved_docs = [f"Error processing: {processed_query}"]
        final_answer = (
            f"Error occurred during complex processing of '{processed_query}'"
        )

    return {"retrieved_docs": retrieved_docs, "final_answer": final_answer}


# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("preprocessing", preprocessing_node)
graph.add_node("simple_processing", simple_processing_node)
graph.add_node("complex_processing", complex_processing_node)

# Add edges
graph.add_edge(START, "preprocessing")
# The preprocessing node handles routing via Command.goto

# Add edges to END from processing nodes
graph.add_edge("simple_processing", END)
graph.add_edge("complex_processing", END)

# Set entry point (this is redundant since we already have START edge, but keeping for clarity)
graph.set_entry_point("preprocessing")

# Compile the graph
agent_runner = graph.compile()

# Example usage
if __name__ == "__main__":
    # Test with simple query
    print("=== Testing Simple Query ===")
    try:
        simple_result = agent_runner.invoke({"user_query": "What is Python?"})
        print("Classification:", simple_result.get("classification"))
        print("Final Answer:", simple_result.get("final_answer"))
        print(
            "Retrieved Docs:", len(simple_result.get("retrieved_docs", [])), "documents"
        )
    except Exception as e:
        print(f"Error in simple query test: {e}")
    print()

    # Test with complex query
    print("=== Testing Complex Query ===")
    try:
        complex_result = agent_runner.invoke(
            {
                "user_query": "Provide a detailed analysis and comparison of machine learning algorithms for complex data processing tasks, including their computational complexity and real-world applications."
            }
        )
        print("Classification:", complex_result.get("classification"))
        print("Final Answer:", complex_result.get("final_answer"))
        print(
            "Retrieved Docs:",
            len(complex_result.get("retrieved_docs", [])),
            "documents",
        )
    except Exception as e:
        print(f"Error in complex query test: {e}")
