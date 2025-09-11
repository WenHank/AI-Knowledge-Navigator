import re


def extract_and_clean_json(text: str) -> str:
    """Extract and clean JSON data from text, handling common issues"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    json_content_match = re.search(r"\[.*\]|\{.*\}", text.strip(), re.DOTALL)
    if json_content_match:
        json_content = json_content_match.group(0)
    else:
        json_content = text

    json_content = re.sub(r",\s*]", "]", json_content)
    json_content = re.sub(r",\s*}", "}", json_content)

    return json_content
