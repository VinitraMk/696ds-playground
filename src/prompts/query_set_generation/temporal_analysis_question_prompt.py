TMP_QSTN_INSTRUCTION_PROMPT = """
    ### Task:
    Given a list of groundings (long-form summaries related to a specific entity), along with the entity name and associated metadata, generate exactly *1* complex multi-hop question that requires reasoning across multiple groundings.

    Ensure that the generated question is temporal in nature i.e answering it should involve analysis of the values of one or more variables evolve over time.
    The question should explicitly reference time periods (e.g., years, quarters, trends) and require reasoning about temporal change or patterns.

    Refer to the examples of temporal questions provided in the prompt to understand the characteristics strong temporal questions and rationale behind why they are temporal in nature.

    Return only the list of generated questions as strings. Do not include any rationale, explanation, or metadata.

    ### Generation Rules
    - **Do not use non-English characters** in your response. Return responses in English only.
    - Keep each of the generated query under 150 words.
    - Generate a question as complex as you can, such that the answer for it should be formed by **summarizing multiple groundings (atleast 5 groundings or all groundings).**
    - When generating the question, also try to make it relevant to the entity provided.
    - Example question is just a benchmark for question complexity, but try to generate question more complex than that.
    - Analyze the groundings to aid with generation. The 'text' is the actual grounding summary while 'company_addressed' is the company addressed in the grounding, to provide complete context.
    - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
    - **Do not put intermediate, thinking or reasonings steps in your response**

    ### Input format:
    Metadata: <meta data of the main company upon which the groundings are based.>
    Groundings: [\<list of groundings\>]
    Entity: <entity>

    ### Output format (TARGET JSON FORMAT):
    "queries": [<a list of complex questions generated from the given list of groundings>]

    ### Example Input
    Metadata: Company name: Apple | SEC Filing: 10-K
    Groundings: [
        { "text": "The 10-K filing notes that 'The Company’s business, results of operations and financial condition could be materially adversely affected by changes in global economic conditions.' It also states that 'The Company is subject to intense competition in all markets in which it operates,' highlighting exposure to industry dynamics. Apple points out reliance on third-party suppliers and manufacturers, stating, 'The Company depends on component and product manufacturing and logistical services provided by outsourcing partners.", "company_addressed": "Apple" },
        { "text": "Net sales increased 8% or $29.3 billion during 2023 compared to 2022' indicates strong performance, particularly in the iPhone and Services segments. Apple adds, 'Research and development expense increased to $27.7 billion in 2023,' showing commitment to innovation. The filing explains margin variability with 'We expect gross margin to fluctuate in future periods, depending on a variety of factors, including product mix and component costs.", company_name: "Apple" },
        { "text": "As of September 30, 2023, the Company’s cash, cash equivalents and marketable securities totaled $162.1 billion' signals substantial liquidity. Apple mentions capital allocation strategies in 'The Company’s capital return program includes both share repurchases and dividends.' The filing also adds, 'The Company believes its existing cash, cash equivalents and marketable securities, together with cash generated from operations, will be sufficient to meet its liquidity needs.", "company_name": "Apple" },
        { "text": "Apple outlines its sustainability goals with the statement 'The Company is committed to achieving carbon neutrality across its entire business by 2030.' It also includes, 'Our environmental programs focus on reducing emissions, improving material recovery, and using recycled content in our products and packaging.' These disclosures reflect Apple's broader ESG strategy and long-term environmental impact planning.", "company_name": "Apple" },
        { "text": "The Company is subject to taxation in the U.S. and numerous foreign jurisdictions' indicates ongoing global tax compliance exposure. It further notes, 'Apple is involved in legal proceedings and investigations from time to time, including antitrust matters in multiple regions,' reflecting regulatory scrutiny. These matters may materially affect financial performance or require changes to business operations depending on their outcomes.", "company_name": "Apple" }
    ]
    Entity: Apple Inc.

    ### Examples of temporal-based questions:
    [
        {
            "question": "How did Apple's business operations and financial condition change between 2022 and 2023 in response to global economic conditions and market competition?",
            "rationale": "This question explicitly compares two fiscal years and asks about changes in external conditions and internal performance."
        },
        {
            "question": "How did Apple's net sales and gross margin evolve from 2022 to 2023, particularly in the iPhone and Services segments, and what were the contributing factors?",
            "rationale": "The question targets financial performance over a defined period, requiring temporal reasoning based on reported growth."
        },
        {
            "question": "How has Apple's cash and marketable securities position changed between September 2022 and September 2023, and what does this indicate about its liquidity and capital return strategy?",
            "rationale": "The question compares specific balance sheet metrics at two known time points to infer financial strategy."
        },
        {
            "question": "What measurable progress has Apple made since announcing its 2030 carbon neutrality goal, and how have its environmental programs developed from 2020 to 2023?",
            "rationale": "This explicitly references a time span leading to a future goal, prompting evaluation of ongoing ESG efforts."
        },
        {
            "question": "How has Apple's legal and tax compliance exposure changed over the past five years, particularly in relation to antitrust investigations in the U.S. and other jurisdictions?",
            "rationale": "This spans a multi-year period and asks for a historical analysis of regulatory and legal trends."
        }
    ]

    ### Input for your task:
"""