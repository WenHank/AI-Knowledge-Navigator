import os
import json
from dotenv import load_dotenv
import requests

from agents.base import BaseAgent
from agents.extract_json import extract_and_clean_json

load_dotenv()


class LocalAgent(BaseAgent):
    def __init__(self):
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.ANSWERING_LOCAL_MODEL_NAME = os.getenv(
            "ANSWERING_LOCAL_MODEL_NAME", "mistralai/mistral-7b-instruct:free"
        )
        self.OPENROUTER_HOST = "https://openrouter.ai/api/v1/chat/completions"

    def run(self, state: dict) -> dict:
        user_query = state.get("user_query", "")

        try:
            response = requests.post(
                url=self.OPENROUTER_HOST,
                headers={
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "model": self.ANSWERING_LOCAL_MODEL_NAME,
                        "messages": [{"role": "user", "content": user_query}],
                    }
                ),
            )

            if response.status_code == 401:
                print("Error: Invalid API key or unauthorized access")
                print("Please check your OPENROUTER_API_KEY in the .env file")
                state["final_answer"] = {"error": "Authentication failed"}
                return state

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("Extracted content:", content)

                # Store the raw content as final answer
                state["final_answer"] = content

                # Also store token usage if available
                if "usage" in response_data:
                    state["token_usage"] = response_data["usage"]
            else:
                print("Error: No choices in API response")
                state["final_answer"] = {"error": "No response content"}

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            state["final_answer"] = {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            state["final_answer"] = {"error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            print(f"Unexpected error: {e}")
            state["final_answer"] = {"error": f"Unexpected error: {str(e)}"}

        return state

    def __call__(self, state: dict) -> dict:
        """Make the agent callable for LangGraph compatibility"""
        return self.run(state)


if __name__ == "__main__":
    processing = LocalAgent()

    state = {"user_query": "What is machine learning?"}
    result_state = processing.run(state)

    print("\n--- Final OpenRouter Agent Result ---")
    print(json.dumps(result_state, indent=2, ensure_ascii=False))
