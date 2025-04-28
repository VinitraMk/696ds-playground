import json
import re

def extract_json_text_by_key(raw_text, target_key):
    """
    Searches raw text for a JSON object that contains a specific key
    and returns it as a Python dictionary. Returns None if not found.
    """
    # Match any JSON object containing the key: { "target_key": "some_value" }
    pattern = rf'\{{[^{{}}]*"{re.escape(target_key)}"\s*:\s*"[^"{{}}]*"[^{{}}]*\}}'

    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def extract_json_array_by_key(raw_text, target_key):
    """
    Extracts a list of strings from a JSON-like structure in LLM output
    where the given key maps to a string array.

    Returns a list of strings, or None if not found or not valid.
    """
    # Regex to find: "key": [ "value1", "value2", ... ]
    pattern = rf'"{re.escape(target_key)}"\s*:\s*\[(?:\s*"[^"]*"\s*,?\s*)+\]'
    #another_pattern = rf'"{re.escape(target_key)}"\s*:\s*\[(.*?)\]'

    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        json_fragment = "{" + match.group(0) + "}"
        try:
            parsed = json.loads(json_fragment)
            return parsed[target_key]
        except json.JSONDecodeError:
            return None
    return None

def extract_json_object_array_by_keys(text, required_keys):
    """
    Extracts a JSON array of objects containing the specified two keys from a text block.
    
    Args:
        text (str): LLM response text that may contain a JSON array.
        required_keys (list or tuple): Two keys expected in each object (e.g., ["factoid", "citation"]).
    
    Returns:
        list: A list of matching dictionaries or an empty list if not found or invalid.
    """
    if len(required_keys) != 2:
        raise ValueError("Exactly two keys must be specified.")

    try:
        # Match JSON array in the text
        json_match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
        if not json_match:
            return []

        json_str = json_match.group(0)
        data = json.loads(json_str)

        # Validate structure: list of dicts with required keys
        if isinstance(data, list) and all(
            isinstance(item, dict) and all(k in item for k in required_keys)
            for item in data
        ):
            return data
        else:
            return []

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

def is_valid_sentence(sentence, word_count_limit = 100):

    sentence = sentence.strip()

    if sentence == 'None' or sentence == "":
        return False
    
    if re.search(r'[\u4e00-\u9fff]', sentence):  # Unicode range for Chinese characters
        return False
    
    if len(sentence) > 10 and " " not in sentence:  # remove strings with no spaces
        return False
    
    if re.search(r'_[a-zA-Z]', sentence):  # Detect underscores replacing spaces
        return False
    
    if len(sentence.split(" ")) > word_count_limit or len(sentence.split(" ")) < 7:
        return False
    
    return True
