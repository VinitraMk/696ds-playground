CMP_QSTN_INSTRUCTION_PROMPT = """
    ### Task:
    You are given the following inputs:
    - A list of groundings (long-form summaries related to a specific entity)
    - The target entity
    - Associated metadata
    Your task is to generate 5 complex, multi-hop questions that require reasoning across multiple groundings. Each question must be comparison-based, meaning it should involve strong and direction comparison of subjects, topics, or entities.

    Refer to the example of comparison-based questions in the prompt to understand the rationale behind effective comparison-based questions.

    Return only the list of generated questions as strings. Do not include any rationale, explanation, or metadata.

    ### Generation Rules
    - **Do not use non-English characters** in your response. Return responses in English only.
    - Keep each of the generated query under 150 words.
    - Make the question as complex as you can, requiring reasoning over multiple (or all) groundings.
    - Generate the question, such that the answer for it should be formed by **summarizing multiple groundings (atleast 5 groundings or all groundings).**
    - When generating the question, also try to make it relevant to the entity provided.
    - Example question is just a benchmark for question complexity, but try to generate question more complex than that.
    - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
    - **Do not put intermediate, thinking or reasonings steps in your response**
    - Use the example structure to return the final response. **Do not copy example from the prompt** in your response.

    ### Input format:
    Metadata: <meta data of the main company upon which the groundings are based.>
    Groundings: [\<list of groundings\>]
    Entity: <entity>

    ### Output format (TARGET JSON FORMAT):
    "queries": [<a list strings consisting of complex questions generated from the given list of groundings>]

    ### Example Input

    Metadata: Company name: Apple | SEC Filing: 10-K
    Groundings: [
        "The 10-K filing notes that 'The Company’s business, results of operations and financial condition could be materially adversely affected by changes in global economic conditions.' It also states that 'The Company is subject to intense competition in all markets in which it operates,' highlighting exposure to industry dynamics. Apple points out reliance on third-party suppliers and manufacturers, stating, 'The Company depends on component and product manufacturing and logistical services provided by outsourcing partners.",
        "Net sales increased 8% or $29.3 billion during 2023 compared to 2022' indicates strong performance, particularly in the iPhone and Services segments. Apple adds, 'Research and development expense increased to $27.7 billion in 2023,' showing commitment to innovation. The filing explains margin variability with 'We expect gross margin to fluctuate in future periods, depending on a variety of factors, including product mix and component costs.",
        "As of September 30, 2023, the Company’s cash, cash equivalents and marketable securities totaled $162.1 billion' signals substantial liquidity. Apple mentions capital allocation strategies in 'The Company’s capital return program includes both share repurchases and dividends.' The filing also adds, 'The Company believes its existing cash, cash equivalents and marketable securities, together with cash generated from operations, will be sufficient to meet its liquidity needs.",
        "Apple outlines its sustainability goals with the statement 'The Company is committed to achieving carbon neutrality across its entire business by 2030.' It also includes, 'Our environmental programs focus on reducing emissions, improving material recovery, and using recycled content in our products and packaging.' These disclosures reflect Apple's broader ESG strategy and long-term environmental impact planning.",
        "The Company is subject to taxation in the U.S. and numerous foreign jurisdictions' indicates ongoing global tax compliance exposure. It further notes, 'Apple is involved in legal proceedings and investigations from time to time, including antitrust matters in multiple regions,' reflecting regulatory scrutiny. These matters may materially affect financial performance or require changes to business operations depending on their outcomes."
    ]
    Entity: Apple Inc.

    ### Examples of comparison-based queries:

    "queries": [
        {
            "question": "How does Apple’s reported financial performance in 2023 compare to the potential risks identified in its 10-K, such as economic uncertainty and competition?",
            "rationale": "Juxtaposes Apple’s strong sales and innovation spending with risks from macroeconomic changes and competitive pressures to assess business resilience.",
        },
        {
            "question": "In what ways does Apple’s revenue growth in iPhone and Services segments compare to its rising R&D expenditures, and what does this suggest about its strategic investment priorities?",
            "rationale": "Here, sales performance is contrasted with innovation spending, both cited in the filings. This prompts reasoning about whether investment is keeping pace with revenue growth or being front-loaded for future returns."
        },
        {
            "question": "How does Apple’s gross margin variability compare across hardware product categories versus software and services, considering component costs and product mix?",
            "rationale": "The question invites comparison between hardware vs. services profitability, using the filing's mention of fluctuating margins and component costs. It pushes for an evaluation of structural differences in Apple's business segments."
        },
        {
            "question": "In what ways do Apple’s environmental impact strategies differ from its legal risk mitigation strategies, particularly in how they shape public perception and regulatory engagement?",
            "rationale": "This compares two non-financial strategies—one proactive (ESG), the other reactive (legal compliance)—both of which affect brand image and long-term risk posture."
        },
        {
            "question": "How does Apple’s use of recycled content and emissions reduction compare to its historical practices in product design and material sourcing?",
            "rationale": "This encourages a temporal comparison between past vs. current sustainability practices, grounded in the company’s disclosures about recycled materials and emissions."
        }
    ]

    ### Input for your task:
"""