NUM_QSTN_INSTRUCTION_PROMPT = """
    ### Task:
    Given a list of groundings (long-form summaries related to a specific entity), along with the entity name and associated metadata, generate exactly *1* complex multi-hop question that requires reasoning across multiple groundings.

    Ensure that the question involves numerical analysis i.e answering it requires computational or quantitative reasoning over multiple numerical variables.
    These should focus on evaluating amounts, rates, percentages, fluctuations, or other measurable quantities.

    Refer to the examples of questions provided in the prompt to understand the characteristics numerical questions and rationale behind why they are of questions involving numerical analysis.

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

    ### Examples of numerical analysis queries:
    [
        {
            "question": "By how much did Apple's net sales increase in absolute and percentage terms in 2023 compared to 2022, and what segments contributed most to this growth?",
            "rationale": "This question requires extracting absolute and percentage change in net sales using provided financial figures, and attributing the growth to specific business segments, which involves numerical comparison and breakdown."
        },
        {
            "question": "How does Apple's 2023 research and development spending of $27.7 billion compare to its net sales growth, and what does this imply about its innovation investment intensity?",
            "rationale": "This question draws a relationship between numerical values—R&D spending and net sales—requiring interpretation of proportional investment strategy and intensity, which is inherently quantitative."
        },
        {
            "question": "What proportion of Apple’s total liquid assets (cash, cash equivalents, and marketable securities) as of September 30, 2023, is represented by its capital return programs, such as share repurchases and dividends?",
            "rationale": "The question asks for a ratio or percentage derived from liquid asset figures and capital allocation strategies, calling for quantitative reasoning over financial components."
        },
        {
            "question": "How does the variability in Apple’s gross margin relate to changes in product mix and component costs, and what numerical trends have been observed over recent periods?",
            "rationale": "This involves understanding margin fluctuation in response to numeric changes in product mix or cost components. Answering it involves analyzing variation over time and interpreting contributing variables."
        },
        {
            "question": "What is the estimated impact of global economic fluctuations on Apple's financial condition based on reported sales growth, margin sensitivity, and liquidity reserves?",
            "rationale": "The question combines numeric evidence like sales growth and margin fluctuations to infer the company’s economic resilience, requiring integration of multiple figures into a broader quantitative assessment."
        }   
    ]

    ### Input for your task:
"""
