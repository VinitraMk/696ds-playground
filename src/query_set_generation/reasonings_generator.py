import os
import torch
from vllm import LLM
import multiprocessing
import numpy as np
import json
from time import time
import sys
import argparse
import gc
from google import genai

from utils.string_utils import extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task

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
    "meta-llama/Meta-Llama-3-70B",
    "gemini-2.0-flash"
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

class ReasoningsGenerator:

    def __init__(self, filename, model_index = 6):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        self.model_name = MODELS[model_index]
        with open("./config.json", "r") as fp:
            cfg = json.load(fp)

        if "QwQ" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                    quantization = "awq",
                    download_dir = HF_CACHE_DIR,
                    max_model_len = 2048 * 4,
                    #gpu_memory_utilization=0.95,
                    tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        elif "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        elif "gemini" in self.model_name:
            self.llm = genai.Client(
                api_key = cfg["google_api_keys"]["vinitramk1"]
            )
            self.model_folder = "gemini"
        elif "Llama" in self.model_name:
            #mf = self.model_name.split("/")[1]
            self.__quantize_llama(self.model_name)
            self.llm = LLM(model=f"./models/llama/{self.model_name}-awq",
                quantization="awq",
                gpu_memory_utilization=0.8,
                download_dir = f'./models/llama/{self.model_name}-awq',
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "llama"
        else:
            raise ValueError('Invalid model index passed!')

    def __generate_reasonings(self, qnag_coll, metadata):

        reasoning_instruction_prompt = """
        ### Task:
        Analyze the provided question and answer pair, set of groundings and the metadata, and generate reasonings for the groundings of the given Q&A pair.
        Reasonings are explainings of why the groundings (citations from original document) support the answer.

        ### Generation Rules
        - **Do not put gibberish, unnecessary, ellaborate adjectives and chinese characters** in your response for either question or the answer.
        - **Do not put opinions, your intermediate reasoning steps used in forming the response**.
        - Every reasoning generated should be concise and under 100 words and in English only.
        - **Do not** generate new factoids to put in the groundings to support the answer.
        - **Do not** put incorrect punctuations in the factoids.
        - Use the example structure as reference to return the final response. **Do not copy example from the prompt** in your response.
        - Return clean reasonings with no typos, grammatical mistakes or erronuous punctuations.
        - **Don't think** for more than 2000 tokens

        ### Input format:
        Question: <question text>
        Answer: <answer text>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Groundings: [\<list of factoids\>]

        ### Output format:
        "reasonings": [\<list of factoids picked supporting the answer\>]

        ### Example Input:
        Question: How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?
        Answer: Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
        Metdata: Company: Apple | SEC-filing: 10-K | Related Topic: Risk Factors and Challenges
        Groundings: [
            "Apple has committed to achieving carbon neutrality across its entire business, including supply chain and product life cycle, by 2030.",
            "Apple invests in renewable energy and low-carbon manufacturing processes as part of its environmental sustainability goals.",
            "The company requires its suppliers to comply with its environmental standards, including carbon reduction initiatives.",
            "Apple works with suppliers to transition to clean energy and energy-efficient production methods.",
            "The company integrates recycled and sustainable materials into product design and manufacturing.",
            "Apple acknowledges that its environmental initiatives may lead to higher costs in the short term due to increased material and compliance expenses.",
            "The company anticipates that its ESG efforts will improve brand reputation and customer loyalty.",
            "Apple views its leadership in ESG initiatives as a competitive advantage, positioning it to mitigate future regulatory and environmental risks."
        ]

        ### Example Output (JSON):
        "reasonings": [
            "This grounding sets the foundation of the answer, as it directly reflects the core commitment Apple made—carbon neutrality across its business by 2030. This long-term strategic goal drives the operational, financial, and supplier-level changes discussed in the answer.",
            "Investing in renewable energy and low-carbon manufacturing directly increases Apple's upfront costs, which explains the immediate impact on its cost structure as noted in the answer.",
            "Requiring suppliers to comply with environmental standards introduces compliance costs and complexity across the supply chain, reinforcing the answer’s point about rising supplier-related expenses and risks.",
            "Collaborating with suppliers on clean energy adoption supports long-term operational efficiencies and cost savings, aligning with the answer's assertion that energy-efficient practices may reduce future operational costs.",
            "The use of recycled and sustainable materials increases product development costs due to design and processing complexity, which directly supports the answer’s explanation of how sustainability initiatives raise short-term expenses.",
            "This grounding validates the answer’s claim that ESG initiatives increase short-term costs, particularly due to the need for new materials and compliance requirements, thus reinforcing the financial trade-offs involved.",
            "Improved brand reputation and customer loyalty contribute to long-term profitability, as mentioned in the answer, especially among eco-conscious consumers willing to pay premium prices.",
            "By viewing ESG leadership as a competitive advantage, Apple positions itself to navigate future regulatory and environmental challenges, which supports the answer’s framing of long-term strategic benefits and risk mitigation."
        ]
        
        ### Input for your task:
        """

        reasoning_instruction_prompts = [reasoning_instruction_prompt + f"\nQuery: {qo['query']}\nAnswer: {qo['answer']}\nMetadata: {metadata}\nGroundings: {qo['groundings']}" for qo in qnag_coll]
        query_sets = []
        missed_sets = []
        if "gemini" in self.model_name:
            for ri, reasoning_instruction_prompt in enumerate(reasoning_instruction_prompts):
                rsummary = execute_gemini_LLM_task(self.llm, reasoning_instruction_prompt)
                print(f'generated response for question: ', rsummary)
                rjson_arr = extract_json_array_by_key(rsummary, "reasonings")
                if rjson_arr != None and len(rjson_arr) > 0:
                    query_sets.append({
                        "query": qnag_coll[ri]["query"],
                        "answer": qnag_coll[ri]["answer"],
                        "reasonings": rjson_arr,
                        "groundings": qnag_coll[ri]["groundings"]
                    })
                else:
                    missed_sets.append({
                        "query": qnag_coll[ri]["query"],
                        "answer": qnag_coll[ri]["answer"],
                        "groundings": qnag_coll[ri]["groundings"]
                    })
        else:
            reasoning_system_prompt = "You are a helpful assistant that given a Q&A pair and groundings, returns reasonings (explanation of why every grounding supports the answer)."
            reasoning_prompt_tokens = [get_prompt_token(rip, reasoning_system_prompt, self.model_name) for rip in reasoning_instruction_prompts]
            routputs = execute_LLM_tasks(self.llm, reasoning_prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)

            for zqf, o in zip(qnag_coll, routputs):
                rsummary = o.outputs[0].text.strip()
                print(f'generated response for question: ', rsummary)
                rjson_arr = extract_json_array_by_key(rsummary, "reasonings")
                if rjson_arr != None and len(rjson_arr) > 0:
                    query_sets.append({
                        "query": zqf["query"],
                        "answer": zqf["answer"],
                        "reasonings": rjson_arr,
                        "groundings": zqf["groundings"]
                    })
                else:
                    missed_sets.append({
                        "query": zqf["query"],
                        "answer": zqf["answer"],
                        "groundings": zqf["groundings"]
                    })

        print('no of valid query sets', len(query_sets))
        print('no of invalid query sets', len(missed_sets))
        return query_sets, missed_sets
        
    def generate_reasonings(self, topic_index = 0):
        all_resp = []

        in_query_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
        main_query_store_fp = f'data/queries/{self.model_folder}/{self.filename}_gen_queries.json'
        
        if os.path.exists(in_query_store_fp):
            with open(in_query_store_fp, 'r') as fp:
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
            print('\nStarting reasoning generation for batch of factoids\n')
            for bi, i in enumerate(range(0, len(filtered_queries), PROMPT_BATCH_SIZE)):
                qnag_coll = [{ 'query': qs["query"], 'answer': qs['answer'], 'groundings': qs['groundings'] } for qs in filtered_queries[i:(i+PROMPT_BATCH_SIZE)]]
                print(f'\nRunning reasoning generation for qna and groundings batch {bi}')
                qobjs, missed_qnag = self.__generate_reasonings(qnag_coll, metadata)
                all_resp.extend(qobjs)
                attempts = 0
                while (len(missed_qnag) != 0) and (attempts < NO_OF_TRIALS):
                    print(f'\nRunning reasoning generation for qna and groundings batch {bi}')
                    qnag_coll = [{ 'query': qs["query"], 'answer': qs['answer'], 'groundings': qs['groundings'] } for qs in missed_qnag]
                    qobjs, missed_qnag = self.__generate_reasonings(qnag_coll, metadata)
                    all_resp.extend(qobjs)
                    attempts+=1

            print('No of valid whole set: ', len(all_resp))
            if len(all_resp) > 0:
                if os.path.exists(main_query_store_fp):
                    with open(main_query_store_fp, 'r') as fp:
                        queries = json.load(fp)
                else:
                    queries = { 'queries': [] }
                topic_queries = [tq for tq in queries["queries"] if tq["topic"] == SEED_METADATA_TOPICS[topic_index]]
                #print("topic queries", topic_queries, len(topic_queries), query_dict)
                if len(topic_queries) > 0:
                    topic_queries = topic_queries[0]
                    topic_queries["query_sets"] = topic_queries["query_sets"] + all_resp
                    for iq,_ in enumerate(queries["queries"]):
                        if queries["queries"][iq]["topic"] == SEED_METADATA_TOPICS[topic_index]:
                            queries["queries"][iq]["query_sets"] = topic_queries["query_sets"]
                else:
                    topic_queries = { "topic": SEED_METADATA_TOPICS[topic_index], "query_sets": [] }
                    topic_queries["query_sets"] = all_resp
                    queries["queries"].append(topic_queries)

                with open(main_query_store_fp, 'w') as fp:
                    json.dump(queries, fp)
        
        else:
            print('Chunk store not found!')
            SystemExit()

    def destroy(self):
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-reasonings-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    reason_gen = ReasoningsGenerator(filename = args.filename, model_index = args.model_index)

    if args.topic_index == -1:
        for ti in range(len(SEED_METADATA_TOPICS)):
            print(f'Generating reasonings for QNA and grounding pair on topic: {SEED_METADATA_TOPICS[ti]}')
            reason_gen.generate_reasonings(topic_index = ti)
            print(f'Finished generating reasonings for topic {SEED_METADATA_TOPICS[ti]}')
    else:
        print(f'Generating reasonings for QNA and grounding pair on topic: {SEED_METADATA_TOPICS[args.topic_index]}')
        reason_gen.generate_reasonings(topic_index = args.topic_index)
        print(f'Finished generating reasonings for topic {SEED_METADATA_TOPICS[args.topic_index]}')

    #reason_gen.destroy()
    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

    # remove intermediate data file
    if os.path.exists(f'intermediate_data/query_sets/{reason_gen.model_folder}/{args.filename}_gen_queries.json'):
        os.remove(f'intermediate_data/query_sets/{reason_gen.model_folder}/{args.filename}_gen_queries.json')
