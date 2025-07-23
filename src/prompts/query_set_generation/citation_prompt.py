CITATION_INSTRUCTION_PROMPT = """
    ### Task:
    Analyze the provided chunk of text, question-answer pair and the metadata about company about which the Q&A pair is,
    and return the exact sentence or sentences that support the answer to the question.

    ### Answer Generation Rules
    - **Do not put opinions, adjectives, elaborations, gibberish or unnecessary adjectives** in your response.
    - The sentences that support the answer, **must** be present in the given chunk of text.
    - The sentences from the chunk of text **must be complete sentences**.
    - **Do not put non-English characters** in your response. Return responses only in English.
    - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.

    ### Input format:
    Text: <chunk of text from SEC filing document of the company>
    Query: <query text>
    Answer: <answer text>
    Metadata: <meta data of the main company upon which the factoids are based.>

    ### Output format (JSON):
    "citations": [<list of sentences from the text supporting the answer>]

    ### Input for your task:
"""