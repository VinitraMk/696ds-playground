import os
import torch
import multiprocessing
from vllm import LLM
import json
from time import time
import sys
import argparse
from utils.string_utils import extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks

COMPANY_DICT = {
    'INTC': 'Intel Corp.',
    'AMD': 'AMD Inc.',
    'NVDA': 'Nvidia Corp.',
    'TSLA': 'Tesla Inc.',
    'F': 'Ford Motor Company',
    'GM': 'General Motors'
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
PROMPT_BATCH_SIZE = 3
NO_OF_TRIALS = 5

class GroundingsGenerator:

    def __init__(self, filename, model_index = 6):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        self.model_name = MODELS[model_index]
        if "QwQ" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                    quantization = "awq",
                    download_dir = HF_CACHE_DIR,
                    max_model_len = 2048 * 4,
                    #gpu_memory_utilization=0.95,
                    tensor_parallel_size=torch.cuda.device_count())
        elif "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count())
        elif "Llama" in self.model_name:
            #mf = self.model_name.split("/")[1]
            self.__quantize_llama(self.model_name)
            self.llm = LLM(model=f"./models/llama/{self.model_name}-awq",
                quantization="awq",
                gpu_memory_utilization=0.8,
                download_dir = f'./models/llama/{self.model_name}-awq',
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "llama"
        self.model_folder = "qwq"

    def __generate_groundings(self, fact_doc_texts, factoids_arr_batch, qna_pairs, metadata):

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
        "groundings": [\<list of factoids picked supporting the answer\>]

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

        zipped_qna_factoids = list(zip(qna_pairs, fact_doc_texts))
        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nQuery: {qo[0]['query']}\nAnswer: {qo[0]['answer']}\nMetadata: {metadata}\nFactoids: {qo[1]}" for qo in zipped_qna_factoids]
        grounding_system_prompt = "You are a helpful assistant that given a Q&A pair and factoids, returns groundings (factoids supporting the answer)."
        grounding_prompt_tokens = [get_prompt_token(gip, grounding_system_prompt, self.model_name) for gip in grounding_instruction_prompts]
        goutputs = execute_LLM_tasks(self.llm, grounding_prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)

        qna_pairs_gen = []
        missed_qna_pairs = []
        zipped_qsnts_factoids = list(zip(qna_pairs, factoids_arr_batch))
        for zqf, o in zip(zipped_qsnts_factoids, goutputs):
            gsummary = o.outputs[0].text.strip()
            print(f'generated response for question: ', gsummary)
            gjson_arr = extract_json_array_by_key(gsummary, "groundings")
            if gjson_arr != None and len(gjson_arr) > 0:
                qna_pairs_gen.append({
                    "query": zqf[0]["query"],
                    "answer": zqf[0]["answer"],
                    "factoids": zqf[1],
                    "groundings": gjson_arr
                })
            else:
                missed_qna_pairs.append({
                    "query": zqf[0]["query"],
                    "answer": zqf[0]["answer"],
                    "factoids": zqf[1],
                })

        print('no of valid qna and grounding pairs', len(qna_pairs_gen))
        print('no of invalid qna and grounding pairs', len(missed_qna_pairs))

        return qna_pairs_gen, missed_qna_pairs
        
    def generate_groundings(self, topic_index = 0):

        all_resp = []

        iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
        
        #chunk_obj = [chunk for chunk in chunk_store["chunks"] if chunk["chunk_filename"] == chunk_fn]
        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of queries formed: ', len(query_arr))
            filtered_queries = [querysets for querysets in query_arr if querysets["topic"] == SEED_METADATA_TOPICS[topic_index]]
            if len(filtered_queries) > 0:
                filtered_queries = filtered_queries[0]["query_sets"]
            else:
                print(f'no queries formed for the topic: {SEED_METADATA_TOPICS[topic_index]}')
                SystemExit()

            print('total length of filtered array: ', len(filtered_queries))
            metadata = f'Company: {self.company_name} | SEC Filing: 10-K | Related topic: {SEED_METADATA_TOPICS[topic_index]}'
            all_resp = []
            attempts = 0
            print('\nStarting grounding generation for batch of factoids\n')
            for bi,i in enumerate(range(0, len(filtered_queries), PROMPT_BATCH_SIZE)):
                qna_pairs = [{ 'query': qs["query"], 'answer': qs['answer'] } for qs in filtered_queries[i:(i+PROMPT_BATCH_SIZE)]]
                factoids_arr_batch = [qs["factoids"] for qs in filtered_queries[i:(i+PROMPT_BATCH_SIZE)]] 
                factoids_doc_batch = ["[" + ",".join(f"\"{item['factoid']}\"" for item in factoid_subarr) + "]" for factoid_subarr in factoids_arr_batch]
                print(f'\nRunning grounding generation for factoids batch {bi}')
                qobjs, missed_qna_pairs = self.__generate_groundings(factoids_doc_batch, factoids_arr_batch, qna_pairs, metadata)
                all_resp.extend(qobjs)
                while (len(missed_qna_pairs) != 0) and (attempts < NO_OF_TRIALS):
                    qna_pairs = [{ 'query': qs["query"], 'answer': qs['answer'] } for qs in missed_qna_pairs]
                    factoids_arr_batch = [qs["factoids"] for qs in missed_qna_pairs] 
                    factoids_doc_batch = ["[" + ",".join(f"\"{item['factoid']}\"" for item in qs['factoids']) + "]" for qs in missed_qna_pairs]
                    print(f'\nRunning grounding generation for factoids batch {bi}')
                    qobjs, missed_qna_pairs = self.__generate_groundings(factoids_doc_batch, factoids_arr_batch, qna_pairs, metadata)
                    all_resp.extend(qobjs)
                    attempts += 1
                    
            print('No of valid query, answer and grounding set: ', len(all_resp))
            if len(all_resp) > 0:
                query_json_path = f'./intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
                queries = { 'queries': [] }
                #print("topic queries", topic_queries, len(topic_queries), query_dict)
                topic_queries = { "topic": SEED_METADATA_TOPICS[topic_index], "query_sets": [] }
                topic_queries["query_sets"] = all_resp
                queries["queries"].append(topic_queries)

                with open(query_json_path, 'w') as fp:
                    json.dump(queries, fp) 
        else:
            print('Chunk store not found!')
            SystemExit()


if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-groundings-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    query_gen = GroundingsGenerator(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(len(SEED_METADATA_TOPICS)):
            print(f'Generating groundings for qna pairs on topic {SEED_METADATA_TOPICS[ti]}')
            query_gen.generate_groundings(topic_index = ti)
            print(f'Finished generating groundings for topic {SEED_METADATA_TOPICS[ti]}')
    else:
        print(f'Generating groundings for qna pairs on topic {SEED_METADATA_TOPICS[args.topic_index]}')
        query_gen.generate_groundings(topic_index = args.topic_index)
        print(f'Finished generating groundings for topic {SEED_METADATA_TOPICS[args.topic_index]}')

    
    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
