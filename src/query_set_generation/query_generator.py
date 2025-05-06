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
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from utils.string_utils import is_valid_sentence, extract_json_text_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer



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
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
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
NO_OF_TRIALS = 3
FILENAMES = [
    '10-K_AMD_20231230',
    '10-K_NVDA_20240128',
    '10-K_F_20231231',
    '10-K_GM_20231231',
    '10-K_INTC_20231230',
    '10-K_TSLA_20231231'
]

class QueryGenerator:

    def __init__(self, model_index = 6):
        #self.filename = filename
        #self.company_name = COMPANY_DICT[filename.split('_')[1]]
        self.device = torch.device("cuda")
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
            self.tokenizer = get_tokenizer(self.model_name)
        elif "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
            self.tokenizer = get_tokenizer(self.model_name)
        elif "gemini" in self.model_name:
            self.llm = genai.Client(
                api_key = cfg["google_api_keys"]["vinitramk1"]
            )
            self.model_folder = "gemini"
        elif "Llama-3.3-70B" in self.model_name:
            model_path = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True,  
                llm_int8_enable_fp32_cpu_offload=True
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,  
                device_map="sequential",  
                offload_folder="/tmp/offload", 
                local_files_only=True
            )
            self.model_folder = "llama"
            self.tokenizer = get_tokenizer(self.model_name)
            #tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        else:
            raise SystemExit('Invalid model index passed!')
        
    def __entity_identifier(self, fact_doc_text, metadata):
        entity_identification_prompt = """
        ### Task:
        Given a list of factoids below, identify important all significant entities or "nouns" described in the factoids.
        This should include but not limited to:
        - Object: Any concrete object that is referenced by the provided content.
        - Organization: Any organization working with the main company either on permanent or temporary basis on some contracts.
        - Concepts: Any significant abstract ideas or themes that are central to the article's discussion.

        ### Input Format:
        - Factoids: [<list of factoids>]

        ### Output Format (JSON):
        "entities": [<list of entities recognized from factoids>]

        ### Input for your task:
        """

    def __generate_queries(self, fact_doc_text, metadata):

        qstn_instruction_prompt = """
        ### Task:
        Given the list of factoids below and metadata, generate a complex question that requires reasoning over multiple factoids

        ### Generation Rules
        - **Do not use chinese characters** in your response. Return responses in English only.
        - Keep the generated query under 100 words.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - **Do not put intermediate, thinking or reasonings steps in your response**
        - Don't think for more than 2000 tokens
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]

        ### Output format:
        Query: {
        "query": <question generated from fact(s) in the given text document>
        }

        ### Example Input
        Metadata: Company name: Apple | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Factoids: {["Apple committed to carbon neutrality across its supply chain by 2030.",
            "Apple sources renewable energy for its global operations.",
            "Apple integrates recycled materials into product design.",
            "The company works with suppliers to reduce emissions.",
            "Upfront costs have increased due to sustainability investments."
        ]}

        ### Example Output:
        Query:
        {
            "query": "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?",
        }

        ### Input for your task:
        """

        qstn_instruction_prompt = qstn_instruction_prompt + f"\nMetadata: {metadata}\nFactoids: {fact_doc_text}"
        qstn_system_prompt = "You are a helpful assistant, that given a list of factoids, generates meaningful and complex questions from it."
        query_strs = []
        if "gemini" in self.model_name:
            qsummary = execute_gemini_LLM_task(self.llm, qstn_instruction_prompt)
            print(f'generated response for question: ', qsummary)
            qjson = extract_json_text_by_key(qsummary, "query")
            if qjson != None and "query" in qjson and is_valid_sentence(qjson["query"], 100):
                query_strs.append(qjson["query"])
        elif "Llama-3.3-70B" in self.model_name:
            qstn_prompt_tokens = self.tokenizer([get_prompt_token(qstn_instruction_prompt, qstn_system_prompt, self.tokenizer)], return_tensors = "pt").to(self.device)
            qoutputs = execute_llama_LLM_task(self.llm, qstn_prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            for j,o in enumerate(qoutputs):
                qsummary = o
                print(f'generated {j}th response for question: ', qsummary)
                if "Input for your task" in qsummary:
                    ti = qsummary.index("Input for your task")
                    qjson = extract_json_text_by_key(qsummary[ti:], "query")
                    print('qjson', qjson)
                    if qjson != None and "query" in qjson and is_valid_sentence(qjson["query"], 100):
                        query_strs.append(qjson["query"])
        else:
            qstn_prompt_tokens = [get_prompt_token(qstn_instruction_prompt, qstn_system_prompt, self.tokenizer)]
            qoutputs = execute_LLM_tasks(self.llm, qstn_prompt_tokens, self.model_name, max_new_tokens=3000, temperature=0.6, top_p=0.9)
            for j, o in enumerate(qoutputs):
                qsummary = o.outputs[0].text.strip()
                print(f'generated {j}th response for question: ', qsummary)
                qjson = extract_json_text_by_key(qsummary, "query")
                if qjson != None and "query" in qjson and is_valid_sentence(qjson["query"], 100):
                    query_strs.append(qjson["query"])

        return query_strs
    
    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        
    def generate_query(self, no_of_qstns = 5, topic_index = 0):

        all_resp = []
        if self.model_folder == "gemini" or self.model_folder == "llama":
            chunk_store_fp = f'data/chunked_data/global_chunk_store/qwq/{self.filename}_chunk_store.json'
        else:
            chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
        
        if os.path.exists(chunk_store_fp):
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_arr = chunk_store["chunks"]
            all_factoids = []
            for chunk in chunk_arr:
                if len(chunk["factoids"]) > 0:
                    all_factoids.extend(chunk["factoids"])
            print('length of the entire facts array: ', len(all_factoids))
            filtered_factoids = [factoid for factoid in all_factoids if factoid["topic"] == SEED_METADATA_TOPICS[topic_index]]
            print('total length of filtered array: ', len(filtered_factoids))
            #factoid_subarr = all_factoids[:MAX_FACTOIDS_TO_SAMPLE]
            metadata = f'Company: {self.company_name} | SEC Filing: 10-K | Related topic: {SEED_METADATA_TOPICS[topic_index]}'
            #for idx in random_indices:
                #factoid_subarr.append(all_factoids[idx])
            all_resp = []
            attempts = 0
            print('\nStarting query generation for batch of factoids\n')
            while (len(all_resp) < no_of_qstns) and (attempts < NO_OF_TRIALS):
                for fbi,i in enumerate(range(0, len(filtered_factoids), MAX_FACTOIDS_TO_SAMPLE)):
                    factoid_subarr = filtered_factoids[i:i+MAX_FACTOIDS_TO_SAMPLE]
                    if len(factoid_subarr) < MIN_FACTOIDS_NEEDED_FOR_GENERATION:
                        factoid_subarr = filtered_factoids
                    factoid_str = "[" + ",\n".join(f"{item['factoid']}" for item in factoid_subarr) + "]"
                    print(f'\nRunning query  generation for factoids batch {fbi}')
                    query_strs = self.__generate_queries(factoid_str, metadata)
                    all_resp.extend([{'query': query_str, 'factoids': factoid_subarr } for query_str in query_strs])
                    print('no of valid qstns formed so far: ', len(all_resp))
                    if len(all_resp) >= no_of_qstns:
                        break
                attempts += 1
                    
            print(f'No of valid queries on topic {SEED_METADATA_TOPICS[topic_index]}: ', len(all_resp))
            if len(all_resp) > 0:
                iquery_json_path = f'./intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
                if os.path.exists(iquery_json_path):
                    with open(iquery_json_path, 'r') as fp:
                        queries = json.load(fp)
                else:
                    queries = { 'queries': [] }
                topic_queries = { "topic": SEED_METADATA_TOPICS[topic_index], "query_sets": [] }
                topic_queries["query_sets"] = all_resp
                queries["queries"].append(topic_queries)

                with open(iquery_json_path, 'w') as fp:
                    json.dump(queries, fp)
        else:
            raise SystemExit('Chunk store not found!')

    def destroy(self):
        print('Destroying llm object')
        gc.collect()
        torch.cuda.empty_cache()
        print('Completed destruction...exiting...')
        os._exit(0)

if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-query-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file


    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 5, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    query_gen = QueryGenerator(model_index = args.model_index)
    print(f'\n\nGenerating queries for file: {args.filename}')
    query_gen.set_filename(args.filename)
    if args.topic_index == -1:
        for ti in range(len(SEED_METADATA_TOPICS)):
            print(f'\nGenerating questions for topic: {SEED_METADATA_TOPICS[ti]}')
            query_gen.generate_query(no_of_qstns = args.no_of_qstns, topic_index = ti)
            print(f'Finished generating questions for topic: {SEED_METADATA_TOPICS[ti]}')
    else:
        print(f'\nGenerating questions for topic: {SEED_METADATA_TOPICS[args.topic_index]}')
        query_gen.generate_query(no_of_qstns = args.no_of_qstns, topic_index = args.topic_index)
        print(f'Finished generating questions for topic: {SEED_METADATA_TOPICS[args.topic_index]}')

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    query_gen.destroy()