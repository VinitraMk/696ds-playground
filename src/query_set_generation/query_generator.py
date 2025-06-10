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
from together import Together

from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer, execute_llama_task_api



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
    "gemini-2.0-flash",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
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
MAX_GROUNDINGS_TO_SAMPLE = 50
MIN_GROUNDINGS_NEEDED_FOR_GENERATION = 15
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
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
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
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            self.llm = Together(api_key = cfg["togetherai_api_key"])
            self.model_folder = "llama"
        else:
            raise SystemExit('Invalid model index passed!')
        
    def __generate_queries_in_single_prompt(self, groundings_doc_text, metadata, entity):

        qstn_instruction_prompt = """
        ### Task:
        Given a list of groundings (sentences related to given entity), entity and metadata, generate 5 complex questions that requires reasoning over multiple groundings.

        ### Generation Rules
        - **Do not use chinese characters** in your response. Return responses in English only.
        - Keep each of the generated query under 100 words.
        - Make the question as complex as you can, requiring reasoning over multiple groundings.
        - Example question is just a benchmark for question complexity, but try to generate question more complex than that.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - **Do not put intermediate, thinking or reasonings steps in your response**
        - Don't think for more than 2000 tokens
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        Metadata: <meta data of the main company upon which the groundings are based.>
        Groundings: [\<list of groundings\>]
        Entity: <entity>

        ### Output format:
        "queries": [<a list of complex questions generated from the given list of groundings>]

        ### Example Input
        Metadata: Company name: Apple | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Groundings: [
            "The 10-K filing notes that 'The Company’s business, results of operations and financial condition could be materially adversely affected by changes in global economic conditions.' It also states that 'The Company is subject to intense competition in all markets in which it operates,' highlighting exposure to industry dynamics. Apple points out reliance on third-party suppliers and manufacturers, stating, 'The Company depends on component and product manufacturing and logistical services provided by outsourcing partners.",
            "Net sales increased 8% or $29.3 billion during 2023 compared to 2022' indicates strong performance, particularly in the iPhone and Services segments. Apple adds, 'Research and development expense increased to $27.7 billion in 2023,' showing commitment to innovation. The filing explains margin variability with 'We expect gross margin to fluctuate in future periods, depending on a variety of factors, including product mix and component costs.",
            "As of September 30, 2023, the Company’s cash, cash equivalents and marketable securities totaled $162.1 billion' signals substantial liquidity. Apple mentions capital allocation strategies in 'The Company’s capital return program includes both share repurchases and dividends.' The filing also adds, 'The Company believes its existing cash, cash equivalents and marketable securities, together with cash generated from operations, will be sufficient to meet its liquidity needs.",
            "Apple outlines its sustainability goals with the statement 'The Company is committed to achieving carbon neutrality across its entire business by 2030.' It also includes, 'Our environmental programs focus on reducing emissions, improving material recovery, and using recycled content in our products and packaging.' These disclosures reflect Apple's broader ESG strategy and long-term environmental impact planning.",
            "The Company is subject to taxation in the U.S. and numerous foreign jurisdictions' indicates ongoing global tax compliance exposure. It further notes, 'Apple is involved in legal proceedings and investigations from time to time, including antitrust matters in multiple regions,' reflecting regulatory scrutiny. These matters may materially affect financial performance or require changes to business operations depending on their outcomes."
        ]
        Entity: Apple Inc.

        ### Example Output:
        "queries": [ "What steps has Apple Inc. taken to manage its financial position and return value to shareholders in fiscal year 2023?" ]

        ### Input for your task:
        """

        qstn_instruction_prompt = qstn_instruction_prompt + f"\nMetadata: {metadata}\nGroundings: {groundings_doc_text}\nEntity: {entity}"
        qstn_system_prompt = "You are a helpful assistant, that given a list of groundings, generates meaningful and complex questions from it."
        query_strs = []
        if "gemini" in self.model_name:
            qsummary = execute_gemini_LLM_task(self.llm, qstn_instruction_prompt)
            print(f'generated response for question: ', qsummary)
            qjson = extract_json_array_by_key(qsummary, "queries")
            if qjson != None and len(qjson) > 0:
                query_strs = [qj for qj in qjson if is_valid_sentence(qj, 100)]
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            qstn_prompt_tokens = self.tokenizer([get_prompt_token(qstn_instruction_prompt, qstn_system_prompt, self.tokenizer)], return_tensors = "pt").to(self.device)
            qoutputs = execute_llama_LLM_task(self.llm, qstn_prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            for j,o in enumerate(qoutputs):
                qsummary = o
                print(f'generated {j}th response for question: ', qsummary)
                if "Input for your task" in qsummary:
                    ti = qsummary.index("Input for your task")
                    qjson = extract_json_array_by_key(qsummary[ti:], "queries")
                    print('qjson', qjson)
                    if qjson != None and len(qjson) > 0:
                        query_strs = [qj for qj in qjson if is_valid_sentence(qj, 100)]
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            qsummary = execute_llama_task_api(self.llm, qstn_instruction_prompt, qstn_system_prompt)
            print('generated response: ', qsummary)
            qjson = extract_json_array_by_key(qsummary, "queries")
            if qjson != None and len(qjson) > 0:
                query_strs = [qj for qj in qjson if is_valid_sentence(qj, 100)]
        else:
            qstn_prompt_tokens = [get_prompt_token(qstn_instruction_prompt, qstn_system_prompt, self.tokenizer)]
            qoutputs = execute_LLM_tasks(self.llm, qstn_prompt_tokens, self.model_name, max_new_tokens=3000, temperature=0.6, top_p=0.9)
            for j, o in enumerate(qoutputs):
                qsummary = o.outputs[0].text.strip()
                print(f'generated {j}th response for question: ', qsummary)
                qjson = extract_json_array_by_key(qsummary, "queries")
                if qjson != None and len(qjson) > 0:
                    query_strs = [qj for qj in qjson if is_valid_sentence(qj)]

        return query_strs
        
    def __generate_queries(self, groundings_doc_text, metadata, entity):

        qstn_instruction_prompt = """
        ### Task:
        Given a list of groundings (sentences related to given entity), entity and metadata, generate a complex question that requires reasoning over multiple groundings.

        ### Generation Rules
        - **Do not use chinese characters** in your response. Return responses in English only.
        - Keep each of the generated query under 100 words.
        - Make the question as complex as you can, requiring reasoning over multiple groundings.
        - Example question is just a benchmark for question complexity, but try to generate question more complex than that.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - **Do not put intermediate, thinking or reasonings steps in your response**
        - Don't think for more than 2000 tokens
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        Metadata: <meta data of the main company upon which the groundings are based.>
        Groundings: [\<list of groundings\>]
        Entity: <entity>

        ### Output format:
        {
            "query": "<a complex question generated from the given list of groundings>"
        }

        ### Example Input
        Metadata: Company name: Apple | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Groundings: [
            "The 10-K filing notes that 'The Company’s business, results of operations and financial condition could be materially adversely affected by changes in global economic conditions.' It also states that 'The Company is subject to intense competition in all markets in which it operates,' highlighting exposure to industry dynamics. Apple points out reliance on third-party suppliers and manufacturers, stating, 'The Company depends on component and product manufacturing and logistical services provided by outsourcing partners.",
            "Net sales increased 8% or $29.3 billion during 2023 compared to 2022' indicates strong performance, particularly in the iPhone and Services segments. Apple adds, 'Research and development expense increased to $27.7 billion in 2023,' showing commitment to innovation. The filing explains margin variability with 'We expect gross margin to fluctuate in future periods, depending on a variety of factors, including product mix and component costs.",
            "As of September 30, 2023, the Company’s cash, cash equivalents and marketable securities totaled $162.1 billion' signals substantial liquidity. Apple mentions capital allocation strategies in 'The Company’s capital return program includes both share repurchases and dividends.' The filing also adds, 'The Company believes its existing cash, cash equivalents and marketable securities, together with cash generated from operations, will be sufficient to meet its liquidity needs.",
            "Apple outlines its sustainability goals with the statement 'The Company is committed to achieving carbon neutrality across its entire business by 2030.' It also includes, 'Our environmental programs focus on reducing emissions, improving material recovery, and using recycled content in our products and packaging.' These disclosures reflect Apple's broader ESG strategy and long-term environmental impact planning.",
            "The Company is subject to taxation in the U.S. and numerous foreign jurisdictions' indicates ongoing global tax compliance exposure. It further notes, 'Apple is involved in legal proceedings and investigations from time to time, including antitrust matters in multiple regions,' reflecting regulatory scrutiny. These matters may materially affect financial performance or require changes to business operations depending on their outcomes."
        ]
        Entity: Apple Inc.

        ### Example Output:
        {
            "query": "What steps has Apple Inc. taken to manage its financial position and return value to shareholders in fiscal year 2023?"
        }

        ### Input for your task:
        """

        qstn_instruction_prompt = qstn_instruction_prompt + f"\nMetadata: {metadata}\nGroundings: {groundings_doc_text}\nEntity: {entity}"
        qstn_system_prompt = "You are a helpful assistant, that given a list of groundings, generates meaningful and complex questions from it."
        query_strs = []
        if "gemini" in self.model_name:
            qsummary = execute_gemini_LLM_task(self.llm, qstn_instruction_prompt)
            print(f'generated response for question: ', qsummary)
            qjson = extract_json_text_by_key(qsummary, "query")
            if qjson != None and "query" in qjson and is_valid_sentence(qjson["query"], 100):
                query_strs.append(qjson["query"])
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
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
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            qsummary = execute_llama_task_api(self.llm, qstn_instruction_prompt, qstn_system_prompt)
            print('generated response: ', qsummary)
            qjson = extract_json_text_by_key(qsummary, "query")
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
        
    def generate_query(self, no_of_qstns = 5, multiple_in_single_prompt = True):

        all_resp = []
        if self.model_folder == "gemini":
            chunk_store_fp = f'data/chunked_data/global_chunk_store/qwq/{self.filename}_chunk_store.json'
            sampled_entities_fp = f'data/chunked_data/global_chunk_store/qwq/{self.filename}_sampled_entities.json'
        else:
            chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
            sampled_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_sampled_entities.json'

        if os.path.exists(chunk_store_fp) and os.path.exists(sampled_entities_fp):
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_arr = chunk_store["chunks"]

            all_groundings = []
            for ci, chunk in enumerate(chunk_arr):
                if "groundings" in chunk and len(chunk["groundings"]) > 0:
                    all_groundings.append({'chunk_index': ci, 'groundings': chunk["groundings"]})

            print('length of the entire groundings array: ', len(all_groundings))

            with open(sampled_entities_fp, 'r') as fp:
                sampled_entities = json.load(fp)['sampled_entities']


            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            all_resp = {}
            print('\nStarting query generation for batch of factoids\n')
            for entity in sampled_entities:
                attempts = 0
                all_resp[entity] = []
                filtered_groundings = []

                for gobj in all_groundings:
                    ci = gobj['chunk_index']
                    rel_groundings = [{'chunk_index': ci, 'entity': entity, 'text': gr['text'] } for gr in gobj['groundings'] if gr['entity'] == entity]
                    filtered_groundings.extend(rel_groundings)


                if multiple_in_single_prompt:
                    for fbi,i in enumerate(range(0, len(filtered_groundings), MAX_GROUNDINGS_TO_SAMPLE)):
                        groundings_subarr = filtered_groundings[i:i+MAX_GROUNDINGS_TO_SAMPLE]
                        if len(groundings_subarr) < MIN_GROUNDINGS_NEEDED_FOR_GENERATION:
                            max_len = max(MAX_GROUNDINGS_TO_SAMPLE, len(filtered_groundings))
                            groundings_subarr = filtered_groundings[-max_len:]
                        chunks_used = list(set([gobj['chunk_index'] for gobj in groundings_subarr]))
                        groundings_str = "[" + ",\n".join(f"{item['text']}" for item in groundings_subarr) + "]"
                        print(f'\nRunning query  generation for factoids batch {fbi}')
                        query_strs = self.__generate_queries_in_single_prompt(groundings_str, metadata, entity)
                        all_resp[entity].extend([{'query': query_str, 'groundings': groundings_subarr, 'chunks_used': chunks_used } for query_str in query_strs])
                        print('no of valid qstns formed so far: ', len(all_resp[entity]))
                        if len(all_resp[entity]) >= no_of_qstns:
                            break
                    attempts += 1

                else:
                    #max_query_count = max(1, min(no_of_qstns, int(len(filtered_groundings) / MAX_GROUNDINGS_TO_SAMPLE)))
                    while (len(all_resp[entity]) < no_of_qstns) and (attempts < NO_OF_TRIALS):
                        for fbi,i in enumerate(range(0, len(filtered_groundings), MAX_GROUNDINGS_TO_SAMPLE)):
                            groundings_subarr = filtered_groundings[i:i+MAX_GROUNDINGS_TO_SAMPLE]
                            if len(groundings_subarr) < MIN_GROUNDINGS_NEEDED_FOR_GENERATION:
                                max_len = max(MAX_GROUNDINGS_TO_SAMPLE, len(filtered_groundings))
                                groundings_subarr = filtered_groundings[-max_len:]
                            chunks_used = list(set([gobj['chunk_index'] for gobj in groundings_subarr]))
                            groundings_str = "[" + ",\n".join(f"{item['text']}" for item in groundings_subarr) + "]"
                            print(f'\nRunning query  generation for factoids batch {fbi}')
                            query_strs = self.__generate_queries(groundings_str, metadata, entity)
                            all_resp[entity].extend([{'query': query_str, 'groundings': groundings_subarr, 'chunks_used': chunks_used } for query_str in query_strs])
                            print('no of valid qstns formed so far: ', len(all_resp[entity]))
                            if len(all_resp[entity]) >= no_of_qstns:
                                break
                        attempts += 1
                        
            if all_resp != {}:
                iquery_json_path = f'./intermediate_data/query_sets/{self.model_folder}/{self.filename}_generated_queries.json'
                queries = { 'queries': {} }
                queries["queries"] = all_resp

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
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 5, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    query_gen = QueryGenerator(model_index = args.model_index)
    print(f'\n\nGenerating queries for file: {args.filename}')
    query_gen.set_filename(args.filename)
    query_gen.generate_query(no_of_qstns = args.no_of_qstns)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    query_gen.destroy()