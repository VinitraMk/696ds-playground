QSTN_CLASSIFIER_PROMPT = """
    ### Task
    You are a question classification expert. Your task is to classify each question into one or more of the following categories:
	- Inference: The question requires reasoning beyond explicit facts, such as deducing implications or making judgments.
    - Comparison: The question compares two or more entities, time periods, or concepts.
    - Temporal: The question involves changes, trends, or events over time.
	- None: The question does not clearly fall into any of the above categories.

    For the question:
	Return the category or categories it belongs to (e.g., [“Inference”, “Temporal”]).
    If it belongs to 'None' class the categories array should only have the string 'None'.

    ### Input Format:
    - Question: <question text>

    ### Output Format (JSON):
    "categories": [<list of question categories>]    

    ### Input for your task:
"""

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