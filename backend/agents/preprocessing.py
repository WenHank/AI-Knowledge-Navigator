import os
import json
from dotenv import load_dotenv
import requests

from agents.base import BaseAgent
from agents.prompts import PROCESSING_PROMPT
from agents.extract_json import extract_and_clean_json

load_dotenv()


# TODO change to use the local fine tuned model(llama3 8B)
class PreprocessingAgent(BaseAgent):
    def __init__(self):
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.PREPROCESSING_MODEL_NAME = os.getenv(
            "PREPROCESSING_MODEL_NAME", "meta-llama/llama-3.3-8b-instruct:free"
        )
        self.OPENROUTER_HOST = "https://openrouter.ai/api/v1/chat/completions"
        self.PROCESSING_PROMPT = PROCESSING_PROMPT

    def run(self, state: dict) -> dict:
        user_query = state.get("user_query", "")
        query = self.PROCESSING_PROMPT + "This is the question: " + user_query

        try:
            response = requests.post(
                url=self.OPENROUTER_HOST,
                headers={
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "model": self.PREPROCESSING_MODEL_NAME,
                        "messages": [{"role": "user", "content": query}],
                    }
                ),
            )

            if response.status_code == 401:
                print("Error: Invalid API key or unauthorized access")
                print("Please check your OPENROUTER_API_KEY in the .env file")
                state["preprocessing_result"] = "1"  # Default fallback
                return state

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("Extracted content:", content)

                # Process the content through JSON extractor
                processed_data = extract_and_clean_json(content)
                print(f"Processed data type: {type(processed_data)}")
                print("Processed data:", processed_data)

                # Extract routing type
                if isinstance(processed_data, dict):
                    routing_type = processed_data.get("type", 1)
                else:
                    # If processed_data is still a string, try to parse it
                    try:
                        parsed_data = json.loads(processed_data)
                        routing_type = parsed_data.get("type", 1)
                    except:
                        routing_type = 1  # Default fallback

                state["preprocessing_result"] = str(routing_type)
            else:
                print("Error: No choices in API response")
                state["preprocessing_result"] = "1"  # Default fallback

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            state["preprocessing_result"] = "1"  # Default fallback
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            state["preprocessing_result"] = "1"  # Default fallback
        except Exception as e:
            print(f"Unexpected error: {e}")
            state["preprocessing_result"] = "1"  # Default fallback

        return state

    def __call__(self, state: dict) -> dict:
        """Make the agent callable for LangGraph compatibility"""
        return self.run(state)


if __name__ == "__main__":
    processing = PreprocessingAgent()

    state = {"user_query": "What is machine learning?"}
    result_state = processing.run(state)

    print("\n--- Final Preprocessing Agent Result ---")
    print(json.dumps(result_state, indent=2, ensure_ascii=False))
