ANSWER_INSTRUCTION_PROMPT = """
    ### Task:
    Analyze the provided question, entity, list of groundings relevant to the question and the entity, and the metadata about the groudings and generate an answer to the question.

    ### Answer Generation Rules
    - Generated answer is ONLY from the given groundings, **do not** hallucinate new information or groundings to answer the query.
    - Return the final answer in as much detail as possible, in a json object. Use the example output as reference for structure.
    - Keep the generated answer detailed and clear but keep it under 400 words.
    - Sentences or points of the answer should be returned as **one single string** as response.
    - Analyze the groundings to aid with generation. The 'text' is the actual grounding summary while 'company_addressed' is the company addressed in the grounding, to provide complete context.
    - **Do not put non-English characters** in your response. Return responses only in English.
    - **No opinions, gibberish, adjectives, elaborations or extra details**
    - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
    - **Do not copy example from the prompt** in your response.

    ### Input format:
    Query: <query text>
    Entity: <entity>
    Metadata: <meta data of the main company upon which the factoids are based.>
    Groundings: [\<list of citations relevant to the question and the entity\>]

    ### Output format (JSON):
    Answer: {
        "answer": <answer to the question, generated from groundings in the given text document>
    }

    ### Example Input
    Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?"
    Entity: "Apple Inc."
    Metadata: Companies addressed in the groundings: ["Apple"] | Input source: SEC 10-K Filings
    Groundings: [
        { "text": "We set an ambitious goal — to make our products carbon neutral by 2030, across our entire supply chain and the lifetime energy use of our customers’ devices.", "company_addressed": "Apple" },
        { "text": "Our corporate operations have run on 100% renewable energy since 2018.", "company_addressed": "Apple" },
        { "text": "Apple also praised its continuing work in recycling, and making new components out of recycled materials. In 2023, 56% of cobalt in Apple batteries came from recycled sources, a 2x increase compared to the previous year.", "company_addressed": "Apple" },
        { "text": "Apple is calling on its suppliers to decarbonize operations as the tech giant looks to become carbon neutral by 2030. The company is asking manufacturers to decarbonize Apple-related operations by taking steps such as running on 100% renewable electricity.", "company_addressed": "Apple" }
        { "text": "Apple plans to invest in renewable energy projects for its suppliers, emissions reduction technologies, product redesigns and recycling techniques, sustainable sourcing practices, and carbon removal projects.", "company_addressed": "Apple" }
    ]

    ### Example Output:
    {
        "answer": "Apple Inc.’s commitment to achieving carbon neutrality across its supply chain and product lifecycle by 2030, as described in its 10-K filing, represents a transformative ESG strategy with implications for cost structure, supplier dynamics, and long-term profitability. Apple has set a clear target: “to make our products carbon neutral by 2030, across our entire supply chain and the lifetime energy use of our customers’ devices.” This effort spans not just internal operations—“Our corporate operations have run on 100% renewable energy since 2018”—but also imposes new standards on suppliers. The company is “asking manufacturers to decarbonize Apple-related operations by taking steps such as running on 100% renewable electricity.” This indicates a strategic push to align upstream partners with its sustainability goals. Such measures likely introduce short- to mid-term cost increases due to investments in “renewable energy projects for its suppliers, emissions reduction technologies, product redesigns and recycling techniques, sustainable sourcing practices, and carbon removal projects.” These outlays may elevate Apple’s cost structure temporarily but are framed as necessary for long-term operational resilience and brand differentiation. On the supplier side, Apple’s push for decarbonization could strain relationships with vendors unable or unwilling to transition. However, Apple mitigates this by directly supporting their transition, which may foster tighter, more collaborative partnerships over time. Apple’s recycling efforts, including the use of recycled materials—“In 2023, 56% of cobalt in Apple batteries came from recycled sources”—also reflect cost containment and risk reduction in critical material sourcing, potentially insulating Apple from raw material price volatility. While risks include higher upfront costs and execution challenges across a global supplier base, potential rewards include regulatory compliance, strengthened ESG positioning, long-term cost savings, and enhanced customer loyalty, all of which may support sustained profitability."
    }

    ### Input for your task:
"""

ANSWER_REFINEMENT_INSTRUCTION_PROMPT = """
    ### Task:
    Given a query-answer pair, some metadata, a list groundings, refine the answer
    to the question using the groundings in the input. Also return the list of factoids used to reframe the answer.

    ### Answer Generation Rules
    - **No opinions, adjectives, elaborations or extra details**
    - Keep the final refined answer precise, summarizing the factoids.
    - Newly refined answer is ONLY from the given factoids, **do not** hallucinate new information or factoids to answer the query.
    - Return the final answer as one single concise paragraph of under 200 words in a json object. Use the example output as reference for structure.
    - **Do not put chinese characters** in your response. Return responses only in English.
    - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response.
    - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
    - **Do not copy example from the prompt** in your response.
    - Don't think for more than 3000 tokens.

    ### Input format:
    Query: <query text>
    Answer: <answer text>
    Metadata: <meta data of the main company upon which the factoids are based.>
    Groundings: [\<list of citations relevant to the Q&A pair.\>]

    ### Output format (JSON):
    Answer: {
        "answer": <answer to the question, generated from fact(s) in the given text document>
    }

    ### Example Input
    Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?"
    Answer: "Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs. On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation. From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments."
    Metadata: Companies addressed in the input: ["Apple"] | Input source: SEC 10-K Filings
    Groundings: [
        { "text": "We set an ambitious goal — to make our products carbon neutral by 2030, across our entire supply chain and the lifetime energy use of our customers’ devices.", "company_addressed": "Apple" },
        { "text": "Our corporate operations have run on 100% renewable energy since 2018.", "company_addressed": "Apple" },
        { "text": "Apple also praised its continuing work in recycling, and making new components out of recycled materials. In 2023, 56% of cobalt in Apple batteries came from recycled sources, a 2x increase compared to the previous year.", "company_addressed": "Apple" },
        { "text": "Apple is calling on its suppliers to decarbonize operations as the tech giant looks to become carbon neutral by 2030. The company is asking manufacturers to decarbonize Apple-related operations by taking steps such as running on 100% renewable electricity.", "company_addressed": "Apple" }
        { "text": "Apple plans to invest in renewable energy projects for its suppliers, emissions reduction technologies, product redesigns and recycling techniques, sustainable sourcing practices, and carbon removal projects.", "company_addressed": "Apple" }
    ]

    ### Example Output:
    Answer: {
        "answer": "Apple’s commitment to achieving carbon neutrality across its entire supply chain and products by 2030, as disclosed in its 10-K, has far-reaching implications for its cost structure, supplier relationships, and long-term profitability. The company has made significant sustainability-driven investments, such as sourcing renewable energy for global operations and integrating recycled materials into product design. These efforts have led to increased upfront costs, reflecting capital expenditures on clean energy infrastructure, low-carbon manufacturing, and material innovation. On the supply chain front, Apple works closely with its partners to enforce carbon reduction targets, which introduces higher compliance costs. This can be particularly challenging for smaller or less-resourced suppliers, potentially creating supply chain risks if partners fail to meet Apple’s environmental standards. However, by fostering collaborations with environmentally aligned suppliers, Apple enhances long-term supplier resilience, innovation potential, and operational synergy. From a profitability perspective, while short-term margins may be compressed due to the elevated costs of sustainability initiatives, the long-term financial outlook remains favorable. These initiatives strengthen Apple’s brand equity, appeal to eco-conscious consumers, and support premium pricing. Additionally, early adoption of robust ESG practices positions Apple to mitigate future regulatory risks, such as carbon taxation or emission-based trade restrictions. Thus, despite near-term financial pressures, Apple’s ESG strategy is likely to yield durable competitive and economic advantages over time.",
    }

    ### Input for your task:
"""

ANSWER_JSON_SCHEMA = {
    "type": "json_object",
    "name": "answer_generation",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
            }
        },
        "required": ["answer"],
        "additionalProperties": False
    }
}