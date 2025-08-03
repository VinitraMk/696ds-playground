ENTITY_INSTRUCTION_PROMPT = """
    ### Task
    Given a chunk of text, identify all significant entites or "nouns" described in the text.
    This should include but not limited to:
    - Object: Any concrete object that is referenced by the provided content.
    - Organization: Any organization working with the main company either on permanent or temporary basis on some contracts.
    - Concepts: Any significant abstract ideas or themes that are central to the text.
    - Events: Any significant event that happens across different time periods in the document or some numerical data about a topic that changes over a time period.

    ### Input Format:
    - Text: <text>

    ### Output Format (JSON):
    "entities": ['entity 1', 'entity 2', ...]

    ### Input for your task:
"""