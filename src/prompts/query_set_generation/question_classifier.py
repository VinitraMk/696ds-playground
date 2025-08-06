QUERY_CLASSIFICATION_INSTRUCTION_PROMPT = """
    ### Task
    You are a question classification expert. Your task is to classify each question into one or more of the following categories:
    - temporal_analysis: these are questions that analyze how the values of one or more variables evolve over time. These questions should explicitly reference time periods (e.g., years, quarters, trends) and require reasoning about temporal change or patterns.
    - numerical_analysis: these questions that require computational or quantitative reasoning over multiple numerical variables. These should focus on evaluating amounts, rates, percentages, fluctuations, or other measurable quantities.
    - entity_interaction_analysis: these questions that examine how different entities (e.g., companies, products, partners) relate to or affect one another. These questions should involve retrieving multiple facts and reasoning about the interplay or relationships between these entities.
    - event_interaction_analysis: these questions that explore how events or developments influence each other. These questions should involve understanding causal, sequential, or correlative relationships between events or actions described in the text.
    - summarization: these questions require generating a summary of the key information across multiple pieces of text. These questions should focus on extracting high-level insights or core messages without deep analysis.
	- none: The question does not clearly fall into any of the above categories.

    For the given question return the category or categories it belongs.

    ### Input Format:
    - Question: <question text>

    ### Output Format (JSON):
    "categories": [<list of categories the question belongs to>]    

    ### Input for your task:
"""

QUERY_CLASSIFICATION_SYSTEM_PROMPT = "You are a helpful assistant, that given a query performs a multi-class classification of the query"

QUERY_CLASSIFICATION_JSON_SCHEMA = {
    "type": "json_object",
    "name": "question_classification",
    "schema": {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["categories"],
        "additionalProperties": False
    }
}


# instruct GPT to give definition refineement for 10-K filing
# Temporal analysis - analyze how the values of a set variables change within a specific time period.
# Numerical analysis - computational analysis of numerical data over multiple variables
# Entity interaction analysis -
    # Question of this type requires retrieval of multiple pieces of related
    # information and analysis of how the different entities interact with each other.
# Event interaction analysis -
    # Question of this type requires retrieval of multiple pieces of related
    # information and analysis of how the events affect/influence each other.
# Summarization