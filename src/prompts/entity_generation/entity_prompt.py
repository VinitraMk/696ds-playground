ENTITY_SYSTEM_PROMPT = "You are a helpful assistant, that given a chunk of text, generates entites addressed in the text."

ENTITY_INSTRUCTION_PROMPT = """
    ### Task
    Given a chunk of text, identify all significant entites or "nouns" described in the text.
    This should include but not limited to:
    - Object: Any concrete object that is referenced by the provided content.
    - Organization: Any organization working with the main company either on permanent or temporary basis on some contracts.
    - Concepts: Any significant abstract ideas or themes that are central to the text.
    - Events: Any significant event that happens across different time periods in the document or some numerical data about a topic that changes over a time period.
    If the entities detected are abbreviations, return their full forms not their abbreviations.
    If a full form does not exist, then return the entity in block case.

    ### Input Format:
    - Text: <text>

    ### Output Format (JSON):
    "entities": ['entity 1', 'entity 2', ...]

    ### Input for your task:
"""

ENTITY_JSON_SCHEMA = {
    "type": "json_object",
    "name": "entity_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["entities"],
        "additionalProperties": False
    }
}