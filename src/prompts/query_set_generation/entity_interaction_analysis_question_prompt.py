ETINT_QSTN_INSTRUCTION_PROMPT = """
    ### Task:
    Given a list of groundings (long summaries related to given entity), entity and metadata, generate exactly *1* complex multi-hop question that requires reasoning over multiple groundings.

    Ensure that generated question should involve entity interaction analysis i.e answering it should involving examining how different entities (e.g., companies, products, partners) relate to or affect one another.
    These questions should involve retrieving multiple facts and reasoning about the interplay or relationships between these entities.

    Refer to the example of questions in the prompt to understand the rationale behind why the questions involve entity interaction.

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

    ### Examples of entity interaction analysis queries:
    [
        {
            "question": "How do Apple's third-party suppliers and manufacturers influence its ability to manage product quality, logistics efficiency, and financial outcomes?",
            "rationale": "This question examines the interaction between Apple and its outsourcing partners, analyzing how those entities impact key operational and financial variables."
        },
        {
            "question": "In what ways do Apple’s investments in R&D interact with the performance of its core product segments, such as iPhone and Services, in driving net sales growth?",
            "rationale": "This involves reasoning about the relationship between R&D spending (an internal entity) and external market-facing product categories, requiring multi-entity analysis."
        },
        {
            "question": "How does Apple’s capital return strategy involving share repurchases and dividends interact with its liquidity position and operational cash flows?",
            "rationale": "This question focuses on the interdependence between liquidity-generating entities (cash reserves and operations) and capital-consuming actions (dividends, buybacks)."
        },
        {
            "question": "How do Apple’s sustainability programs interact with its supply chain partners, particularly in reducing emissions and enhancing material recovery?",
            "rationale": "The question targets the dynamic between Apple’s ESG initiatives and its suppliers, asking for how these entities work together toward environmental goals."
        },
        {
            "question": "What is the relationship between Apple’s exposure to antitrust legal proceedings and its competitive positioning in global markets?",
            "rationale": "This explores the interaction between regulatory authorities (legal proceedings) and Apple’s competitive behavior in different jurisdictions, connecting two distinct but interacting entities."
        }
    ]

    ### Input for your task:
"""