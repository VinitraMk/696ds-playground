


GROUNDING_INSTRUCTION_PROMPT = """
    ### Task:
    Analyze the provided chunk of text, the entity and the metadata about the text and generate a detailed 
    summary about sentences from the chunk of text addressing the entity, directly or indirectly. This summary is called a grounding.

    ### Generation Rules
    - **Generate responses in English only **
    - **Do not put gibberish, incorrect punctuations, unnecessary, ellaborate adjectives and non-English characters** in your response.
    - **Do not put opinions, your intermediate reasoning steps used in forming the response**.
    - The groundings can be short or long. The summary of the sentences related to the given entity (grounding) should be detailed and clear, covering
    the necessary background or context around the sentences as well.
    - Generate the groundings in as much detail as possible, but keep each grounding below 200 words.
    - **Do not** generate completely new groundings that are not addressed in the given text.
    - Use the example structure as reference to return the final response. **Do not copy example from the prompt** in your response.


    ### Input format:
    Text: <chunk of text from SEC filing>
    Entity: <entity>
    Metadata: <meta data of the main company from whose SEC 10-K filing the chunk of text is from>


    ### Output format (JSON):
    "groundings": [\<list of sentences, that related to the entity\>]

    ### Example:
    "groundings": [
        "The 10-K filing notes that 'The Company’s business, results of operations and financial condition could be materially adversely affected by changes in global economic conditions.' It also states that 'The Company is subject to intense competition in all markets in which it operates,' highlighting exposure to industry dynamics. Apple points out reliance on third-party suppliers and manufacturers, stating, 'The Company depends on component and product manufacturing and logistical services provided by outsourcing partners.",
        "Net sales increased 8% or $29.3 billion during 2023 compared to 2022' indicates strong performance, particularly in the iPhone and Services segments. Apple adds, 'Research and development expense increased to $27.7 billion in 2023,' showing commitment to innovation. The filing explains margin variability with 'We expect gross margin to fluctuate in future periods, depending on a variety of factors, including product mix and component costs.",
        "As of September 30, 2023, the Company’s cash, cash equivalents and marketable securities totaled $162.1 billion' signals substantial liquidity. Apple mentions capital allocation strategies in 'The Company’s capital return program includes both share repurchases and dividends.' The filing also adds, 'The Company believes its existing cash, cash equivalents and marketable securities, together with cash generated from operations, will be sufficient to meet its liquidity needs.",
        "Apple outlines its sustainability goals with the statement 'The Company is committed to achieving carbon neutrality across its entire business by 2030.' It also includes, 'Our environmental programs focus on reducing emissions, improving material recovery, and using recycled content in our products and packaging.' These disclosures reflect Apple's broader ESG strategy and long-term environmental impact planning.",
        "The Company is subject to taxation in the U.S. and numerous foreign jurisdictions' indicates ongoing global tax compliance exposure. It further notes, 'Apple is involved in legal proceedings and investigations from time to time, including antitrust matters in multiple regions,' reflecting regulatory scrutiny. These matters may materially affect financial performance or require changes to business operations depending on their outcomes."
    ]
    
    ### Input for your task:
"""

GROUNDING_EVALUATION_PROMPT = """
    ### Task

    You are an expert evaluator assessing the quality of entity-specific summarized sentences (called groundings) generated from a given **text chunk, entity and accompanying metadata**.
    Your evaluation must be based on **five binary criteria**. For each criterion, assign a score of `1` (Yes) if the requirement is fully met, or `0` (No) if it is not. After scoring, provide a **brief justification** for the overall evaluation.
    State the reason for the scores must be explained in the "justification".

    ### Evaluation Criteria:

    1. **Entity Relevance**  
    Are the groundings clearly related to the specified entity? Are they focused and specific to that entity?

    2. **Source Faithfulness**  
    Do the groundings faithfully reflect the content of the chunk and metadata, without introducing factual inaccuracies? Minor reasoning or inference is acceptable if logically supported by the source (e.g., summarizing trends or basic financial implications).

    3. **Key Info Coverage**  
    Do the groundings capture the most important, entity-relevant points from the chunk and metadata?

    4. **Numeric Recall**  
    Are critical numbers (e.g., statistics, dates, financial figures) relevant to the entity correctly included in the groundings?

    5. **Non‑Redundancy**  
    Are the groundings concise, avoiding unnecessary repetition or duplication?

    ### Output format (JSON):
    Evaluation: {{
        "evaluation": {{
            "entity_relevance": 0 or 1,
            "source_faithfulness": 0 or 1,
            "key_info_coverage": 0 or 1,
            "numeric_recall": 0 or 1,
            "non_redundancy": 0 or 1,
            "justification": "brief reason"
        }}
    }}

    ### Input for your task:
    **Entity**: {entity}
    **Chunk**:
    {chunk}
    **Metadata** (if any):
    {metadata}
    **Groundings**:
    {groundings}

"""

GROUNDING_REFINEMENT_PROMPT = """
    ### Task:
    Analyze the provided chunk of text, the entity, the metadata about the text and a list of sentences summarizing
    from the chunk of text addressing the entity (groundings). Also consider the evaluation of the evaluation information provided, the individual score
    and the justification. Understand the individual metric scores and justification and refine the groundings
    to improve the individual metric scores and addresses weaknesses explained in the justification.

    ### Generation Rules
    - **Generate responses in English only **
    - **Do not put gibberish, incorrect punctuations, unnecessary, ellaborate adjectives and non-English characters** in your response.
    - **Do not put opinions, your intermediate reasoning steps used in forming the response**.
    - **Do not** generate completely new groundings that are not addressed in the given text.
    - Use the example structure as reference to return the final response. **Do not copy example from the prompt** in your response.


    ### Input format:
    Text: <chunk of text from SEC filing>
    Entity: <entity>
    Metadata: <meta data of the main company from whose SEC 10-K filing the chunk of text is from>
    Groundings: [\<list of sentences that are related to the entity>\]
    Evaluation: <evaluation of the groundings against various criteria such as relevance, source faithfullness>


    ### Output format (JSON):
    "groundings": [\<list of refined sentences, that related to the entity\>]

    ### Input for your task
"""