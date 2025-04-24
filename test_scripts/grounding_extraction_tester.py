import json
import os
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch



def __get_prompt_token(prompt_text):
    tokenizer = AutoTokenizer.from_pretrained(f"./models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    tokenizer.lang = "en"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that given a Q&A pair and factoids, returns groundings (factoids supporting the answer)."},
        {"role": "user", "content": prompt_text}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, tokenizer(prompt)

def __execute_LLM_tasks(prompts, max_new_tokens, temperature = 0.3, top_p = 0.9):
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        stop=["###EOF###"]
    )
    #model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
    print('prompting LLM')
    outputs = llm.generate(prompts, sampling_params)
    #response_text = tokenizer.decode(
        #outputs[0].outputs[0].token_ids,
        #skip_special_tokens=True,
        #clean_up_tokenization_spaces=True  # Ensures proper spacing
    #)
    #return outputs[0].outputs[0].text if outputs else ""
    #print('getting llm response')
    #print('llm response', outputs[0].outputs[0].text)
    return outputs

FACTOIDS_SUBARR = [
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "US Government introduced licensing requirements impactingexports toChina,HongKong,Macau,andRussiaincludingA100andH100integratedcircuitsduringthethirdquarteroffiscalyear2023"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Climate-change responsiveness issues might lead to facing lawsuits"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "We risk failing to meet set environmental targets leading to higher unforeseen expenses."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Supply-chain interruptions caused by climatic events pose contract dispute dangers resulting potentially in elevated court cases along with cost hikes."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Our GPU's substantial power demand raises operational concern even though they're designed efficiently compared against alternatives."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Inability achieving intended outcomes post-acquisition impacts product development adversely affecting overall growth prospects."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Investments carry risks such as distraction of funds and managerial focus."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Aquisition-related risk includes technological assimilation challenge among others."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Merging businesses might lead to higher debts and interests along with covenant obligations satisfaction pressure."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "We derive substantial revenues from few partnerships/distributors making them critical contributors."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "One specific client accounted for 13 percent of overall earnings during FY-2024 under 'Compute&Networking' division."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "A portion of Company\u2019s revenues comes from partnerships."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Customer A contributed 13% towards the firm\u2019s overall income during FY 2024."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "The contribution by Client A specifically pertains to activities under 'Compute & Networking' division."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Audits were performed according to PCAOB standards"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Procedures include assessment of risks leading to potential errors or fraudulent activities within the reports."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Auditing involves examination of transactional evidences selectively."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "The evaluations cover key estimative judgements done during accountancy processes alongwith review of general report presentations."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Critique matters reported involve high complexity requiring special attention."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Gains or losses from foreign currency remeasurement areincludedinearningsinthoseConsolidatedStatementsofincomeandtodatehavenotbeensignificant."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Management\u2019s estimates of fair value involve uncertainty"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "The estimation includes projections regarding amounts and timings of future cashflows."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Adjustment offsets occur within one-year post-acquisitions affecting reported Goodwill initially."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Investment types consist of both publically-traded equities & private company shares without direct trading visibility."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "A measurable reduction occurs when impairing identifiable lesser-liquidity holdings leading towards marked-down assessments impacting earnings reports negatively via Other Expense segment entries."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Up to 37 million shares of company's common stock could be issued via stock incentives."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Under the 2007 plan, 147 million shares remain ungranted but earmarked towards potential new allocations post-January 28th , 2024"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Incentive stocks' time-to-maturity spans between three-four year cycles depending upon type i.e., regular PSU vs performance linked ones including those tied into broader metrics/market conditions; majority follow four-years duration rules barring some special cases where shorter/thrice-a-quarter installment release model applies."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Nvidia\u2019s ESPP allows employee deductions ranging upto max limit set initially defined @ ~quarterly withholding rate limited till board intervenes cutting down said %age participation cap."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "The actual unit prices obtained through these schemes equate roughly around eighty-five percent discount off initial start line reference mark fluctuating contingent valuation changes across subsequent pick-up intervals concluding two biannual segments spread out overall program span lasting twenty fours month tenure altogether since launch inception back early last decade ago circa mid-decade timeframe ie approx near vicinity surrounding fiscal yr starting point dating December twelve onwards ."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Realized gains and losses were insignificant across reported time frames."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Nvidia's total Amorted Cost Jan-28-'24 = \\$25,684 million"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Mature within less-than-one-year Cash Equivalents' Estimated Fair Valuation Date-January/End/'24 : USD\\$16 billion approximating actual valuation difference (\\$.mil)"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Deferred revenue including remainig performace obligation at NVIDIA Corp was \\$1.1 billion as of Januayr 28, 2024."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Approximately 40 percent of Nvidia's reserved revenues scheduled recognition period falls under upcoming twleve month window."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Nvidia uses foreign currecny forwards covering total nominal amounting USD 1,168 million by end-January-2024 intended primarily mitigating operational cost volatility through anticipated forex moves deemed 'cash-flow' oriented protective measures against adverse fluctuations;"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "As per records ending early February both consecutive recent historical observation points (January), cumulative yet-to-realize gain/losses tied their active but non-significantly impactful portfolio consisting cross-currency swap agreements stood minimal."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "$82 million balance at start of Jan 28, 2024"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "$46 million balance at start of Jan 29, 2023"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "$22 million balance at start of Jan 30, 2022"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Added $278M during year ending january 28th , 2024 "
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "...added total amount added over previous periods : ( ) ; utilization amounts utilized across different timeframes (-)$; balances carried forward into next reporting cycle ($). "
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Federal R&D tax credits total ($M): S."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Tax benefit percentage change Jan'28 vs previous year:"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Income Tax Expense (+/-)% Change FY End '24 v/s prev yr:"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "$ Research & Other Credit Carryforward Increase Y/Y:$+X million."
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "NOL/Capital Loss Carried Forward Increased From Previous Year By : $"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Increase In Total Deferred Assets Between Years = "
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "% Decrease Valuation Allowance For Capital Gain/Losses And Others"
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Undist Earnings Reinvested Intent Indefinitely Amount ="
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Determining Unrecog Deferral Liabilities Due Temp Diff Related Investment Impractical="
                        },
                        {
                            "topic": "Risk Factors and Challenges",
                            "factoid": "Valu Allwnc Relatng CapLoss CF,S&O DT Ass Not Likely Realizab ="
                        }
                    ]

COMPANY_DICT = {
    'INTC': 'Intel Corp.',
    'AMD': 'AMD Inc.',
    'NVDA': 'Nvidia Corp.',
    'TSLA': 'Tesla Inc.',
    'F': 'Ford Motor Company'
}

MODELS = [
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    "Qwen/QwQ-32B-AWQ",
    "meta-llama/Meta-Llama-3-70B"
]

SEED_METADATA_TOPICS = [
    "Risk Factors and Challenges",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
    "Market Trends, Economic Environment, and Industry Dynamics"
]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR

RELEVANCE_THRESHOLD = 2.0
MAX_FACTOIDS_TO_SAMPLE = 25
MIN_FACTOIDS_NEEDED_FOR_GENERATION = 15

llm = LLM(model=f"./models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    quantization = "gptq_marlin",
    download_dir = HF_CACHE_DIR,
    max_model_len = 2048 * 4,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=torch.cuda.device_count())
'''
query_dict = {
    "query": "How do climate change concerns intersect with export restrictions on NVIDIA GPUs like the H100 integrated circuits imposed since fiscal year 2023 third quarter, considering their significant electricity consumption could face legal challenges if unmet regulatory compliance leads to additional litigation?",
    "answer": "NVIDIA faces intersection between climate change responses and recent US government-imposed export controls targeting advanced chips including the H100 series sold internationally starting late calendar years aligned with Fiscal Year 2023 quarters three onwards toward China regions plus Hong Kong/Macau/Russia markets amid heightened scrutiny around technology transfers abroad coupled stringent local emission norms enforcement possibly triggering further judicial actions should non-compliance arise especially concerning considerable electrical demands posed despite efficient designs."
}
'''
query_dict = {
    "query": "Given Nvidia Corporation faces significant climate-responsive legal challenges alongside export restrictions on GPUs like H100s integrated circuits imposed recently, how do rising compliance pressures intersect financially considering major customer dependency (~13%) coupled with acquisition integration complexities? Additionally, evaluate if current deferred assets adjustments can offset any projected increase in operating liabilities?",
    "answer": "Rising regulatory compliance-related costs due to climate change and risk of lawsuits could lead to increased expenses for NVDA. NVIDIA underwent recent export restrictions on GPUs like the H100 in Q4 FY'23 affecting supply chains. Additionally, integration complexities from merging businesses can increase operational debts. Interdependence on major customers accounting ~ 13% demand requires careful management. Supply chain disruptions due to climatic factors might escalate risks. A portion deferred assets adjustment maintained but may not fully offset projected increases in operating liabilities."
}

chunk_store_fp = f'data/chunked_data/global_chunk_store/qwq/10-K_NVDA_20240128_chunk_store.json'

cleanup_instruction = """
    ### Task:
    Given a raw LLM response, your task is to extract a coherent JSON object after fixing typos, grammatical mistakes, json formatting errors, etc.
    The LLM response provided in the input, is the raw response from an LLM tasked with return a json array with the structure "groundings": ["factoid 1", "factoid 2"].
    The LLM response is riddled with json formatting errors, chinese characters, typos, grammatical mistakes, etc. Your task is to understand the LLM response and extract
    the required JSON object from the raw response. The response may not strictly have a properly defined JSON object, but erroneous JSON object (for example, missing array closing brackets).

    ### Guidelines:
    - Extract the JSON object from the raw response. Refer to output format given below, for the JSON object to be returned.
    - **Do not** add more thinking steps or reasoning steps in the response.
    - Return a clean JSON object, with grammatical mistakes, typos and formatting errors from the input, fixed.
    - All responses should only be in English. Translate chinese text in the final response.

    ### Input Format:
    Text - <LLM response text>

    ### Output Format (JSON):
    "groundings": [<list of meaningful, grammatically correct sentences>]

    ### Input for your task
    """

grounding_instruction_prompt = """
    ### Task:
    Analyze the provided question and answer pair, set of factoids and the metadata about the factoids, and generate groundings for the question and answer pair.
    Groundings are factoids that support the answer to the question. The factoids don't have to directly support the answer but should help indirectly answering the
    provided question.

    ### Generation Rules
    - **Do not put gibberish, unnecessary, ellaborate adjectives and chinese characters** in your response for either question or the answer.
    - **Do not put opinions, your intermediate reasoning steps used in forming the response**.
    - The groundings should be factoids picked directly from the provided factoids in the input prompt.
    - **Do not** generate new factoids to put in the groundings to support the answer.
    - **Do not** put incorrect punctuations in the factoids.
    - Use the example structure as reference to return the final response. **Do not copy example from the prompt** in your response.
    - Return clean groundings with no typos, grammatical mistakes or erronuous punctuations.
    - Phrase your response as concisely as possible, in English only.
    - **Don't think** for more than 2000 tokens

    ### Input format:
    Question: <question text>
    Answer: <answer text>
    Metadata: <meta data of the main company upon which the factoids are based.>
    Factoids: [\<list of factoids\>]

    ### Output format:
    "groundings": [\<list of factoids from the input, supporting the answer\>]

    ### Example Input:
    Question: How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?
    Answer: Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
    Metdata: Company: Apple | SEC-filing: 10-K | Related Topic: Risk Factors and Challenges
    Factoids: [
        "Apple has committed to achieving carbon neutrality across its entire business, including the supply chain and product life cycle, by 2030.",
        "The company has made significant investments in renewable energy and low-carbon manufacturing technologies.",
        "Apple requires its suppliers to adhere to strict environmental standards, including reducing carbon emissions and using renewable electricity.",
        "Apple’s Supplier Clean Energy Program has contributed to its goal of reducing emissions across the supply chain.",
        "The company reports that its use of recycled and sustainable materials in products is increasing year-over-year.",
        "Apple continues to invest in energy-efficient technologies that improve operational efficiency across its facilities.",
        "The company discusses potential risks in its 10-K filings related to environmental regulation and climate change, including potential costs from future carbon taxes.",
        "Apple acknowledges short-term cost increases related to sustainability efforts but positions them as long-term strategic investments.",
        "The company’s ESG initiatives are framed as important for protecting brand value and aligning with consumer expectations.",
        "Environmental sustainability is identified as a strategic area for long-term growth and innovation in Apple’s regulatory and investor communications.",
        "Apple’s financial risk disclosures related to foreign exchange fluctuations in emerging markets.",
        "Details on litigation or legal contingencies unrelated to environmental policies.",
        "Statements about Apple’s R&D expenditures for chip architecture and performance optimization.",
        "Inventory management practices tied to consumer demand and holiday cycles.",
        "Tax strategies related to international operations and intellectual property.",
        "Descriptions of supply chain risks unrelated to ESG, such as natural disasters or geopolitical tension.",
        "Information about share repurchase programs or dividend policies.",
        "Macroeconomic risks including inflation or interest rate sensitivity."
    ]

    ### Example Output (JSON):
    "groundings": [
        "Apple has committed to achieving carbon neutrality across its entire business, including supply chain and product life cycle, by 2030.",
        "Apple invests in renewable energy and low-carbon manufacturing processes as part of its environmental sustainability goals.",
        "The company requires its suppliers to comply with its environmental standards, including carbon reduction initiatives.",
        "Apple works with suppliers to transition to clean energy and energy-efficient production methods.",
        "The company integrates recycled and sustainable materials into product design and manufacturing.",
        "Apple acknowledges that its environmental initiatives may lead to higher costs in the short term due to increased material and compliance expenses.",
        "The company anticipates that its ESG efforts will improve brand reputation and customer loyalty.",
        "Apple views its leadership in ESG initiatives as a competitive advantage, positioning it to mitigate future regulatory and environmental risks."
    ]
    
    ### Input for your task:
    """

with open(chunk_store_fp, 'r') as fp:

    log_fp = f'logs/bu-query-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file


    chunk_store = json.load(fp)
    chunk_arr = chunk_store["chunks"]
    all_factoids = []
    for chunk in chunk_arr:
        if len(chunk["factoids"]) > 0:
            all_factoids.extend(chunk["factoids"])
    print('length of the facts array: ', len(all_factoids))
    #chunk_topics = ",".join(chunk_obj["topics"])
    #random_indices = random.sample(range(0, len(all_factoids)), MAX_FACTOIDS_TO_SAMPLE)
    filtered_factoids = [factoid for factoid in all_factoids if factoid["topic"] == SEED_METADATA_TOPICS[0]]
    print('total length of filtered array: ', len(filtered_factoids))
    #factoid_subarr = all_factoids[:MAX_FACTOIDS_TO_SAMPLE]
    metadata = f'Company: Nvidia | SEC Filing: 10-K | Related topic: {SEED_METADATA_TOPICS[0]}'
    #for idx in random_indices:
        #factoid_subarr.append(all_factoids[idx])
    all_resp = []
    print('\nStarting query generation for batch of factoids\n')
    for i in range(0, len(filtered_factoids), MAX_FACTOIDS_TO_SAMPLE):
        factoid_subarr = FACTOIDS_SUBARR
        if len(factoid_subarr) < MIN_FACTOIDS_NEEDED_FOR_GENERATION:
            factoid_subarr = FACTOIDS_SUBARR
        factoid_str = "[" + ",".join(f"{item['factoid']}" for item in factoid_subarr) + "]"
        instruction_prompt = grounding_instruction_prompt + f"\nQuestion: {query_dict['query']}\nAnswer: {query_dict['answer']}\nMetadata: {metadata}\nFactoids: {factoid_str}"
        prompt_token = __get_prompt_token(instruction_prompt)
        instruction_prompt_token = [prompt_token[0]]
        print('prompt token len', len(prompt_token[1]["input_ids"]))
        outputs = __execute_LLM_tasks(instruction_prompt_token, max_new_tokens=4096, temperature=0.6, top_p = 0.9)
        #print('grounding raw response', outputs[0].outputs[0].text)
        #print('stripping trailing white spaces')
        gsummary = outputs[0].outputs[0].text.strip()
        print('len of summary', len(gsummary))
        #print('building next prompt')
        instruction_prompt = cleanup_instruction + f"\nText: {gsummary}"
        prompt_token = __get_prompt_token(instruction_prompt)
        instruction_prompt_token = [prompt_token[0]]
        print('prompt token len', len(prompt_token[1]["input_ids"]))
        outputs = __execute_LLM_tasks(instruction_prompt_token, max_new_tokens = 4096, temperature=0.3, top_p = 0.9)
        print('grounding clean response', outputs[0].outputs[0].text)
    #print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()