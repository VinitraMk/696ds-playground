import os
import torch
from vllm import LLM
import multiprocessing
import json
from time import time, sleep
import sys
import argparse
import re
import gc
from google import genai
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from together import Together
from groq import AsyncGroq

from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer, execute_llama_task_api, execute_groq_task_api
from src.prompts.query_set_generation.question_classifier import QSTN_CLASSIFIER_PROMPT
from consts.company_consts import COMPANY_DICT
import consts.consts


class QueryClassifier:

    def __init__(self, model_index = 6, prompt_batch_size = 3):
        self.model_name = MODELS[model_index]
        self.prompt_batch_size = prompt_batch_size

        self.device = torch.device("cuda")
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
                api_key = cfg["google_api_keys"]["vinitramk4"]
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
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            self.llm = Together(api_key = cfg["togetherai_api_key"])
            self.model_folder = "llama"
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            self.llm = AsyncGroq(api_key = cfg["groq_api_key"])
            self.model_folder = "llama"
        else:
            raise SystemExit('Invalid model index passed!')

    def __get_output_from_llm(self, instruction_prompts, system_prompt, json_schema = None, llm_config = None):
        summary = ""
        if self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            prompt_tokens = self.tokenizer([get_prompt_token(instruction_prompts[0], system_prompt, self.tokenizer)], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            outputs = execute_llama_LLM_task(self.llm, prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            summary = outputs[0]
            if "Input for your task" in summary:
                ti = summary.index("Input for your task")
                summary = summary[ti:]
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            summary = execute_llama_task_api(self.llm, instruction_prompts[0], system_prompt)
            print('generated response: ', summary)
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            summary = execute_groq_task_api(self.llm, json_schema, instruction_prompts, system_prompt)
            print('generated response: ', summary[0])
        else:
            prompt_tokens = [get_prompt_token(instruction_prompts[0], system_prompt, self.tokenizer)]
            outputs = execute_LLM_tasks(self.llm, prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)
            summary = outputs[0].outputs[0].text.strip()
            print(f'generated response: ', summary)

        return summary

    def __classify_qstn(self, qg_pair):

        qstn_classification_prompt = QSTN_CLASSIFIER_PROMPT

        qstn_classification_prompt = qstn_classification_prompt + f"\nQuery: {qg_pair['query']}"
        qstn_classification_system_prompt = "You are a helpful assistant, that given a query performs a multi-class classification of the query"
        categories = []
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            qclass_json_schema = {
                "type": "json_object",
                "name": "question_classification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["categories"],
                    "additionalProperties": False
                }
            }
        else:
            qclass_json_schema = None
        
        qclass_summary = self.__get_output_from_llm([qstn_classification_prompt], qstn_classification_system_prompt, qclass_json_schema)
        qclass_json = extract_json_array_by_key(qclass_summary[0], "categories")
        if qclass_json != None and len(qclass_json) > 0:
            categories = list(map(lambda q: QUERY_TYPE_MAP[q], qclass_json))
            
        return categories

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        
    def classify_qstn(self, no_of_entities = 20):
        
        iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_generated_queries.json'
        main_query_store_fp = f'data/queries/{self.model_folder}/{self.filename}_generated_queries.json'

        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of entities formed: ', len(query_arr))

            if os.path.exists(main_query_store_fp):
                with open(main_query_store_fp, 'r') as fp:
                    main_query_store = json.load(fp)
            else:
                main_query_store = { "queries": {} }
            #chunk_topics = ",".join(chunk_obj["topics"])
            #random_indices = random.sample(range(0, len(all_factoids)), MAX_FACTOIDS_TO_SAMPLE)

            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting answer generation for batch of questions\n')
            sampled_entities = list(query_arr.keys())
            for ei in range(no_of_entities):
                entity = sampled_entities[ei]
                filtered_queries = query_arr[entity]
                #filtered_queries = [qobj for qobj in filtered_queries if "answer" not in qobj]
                for qi, query_obj in enumerate(filtered_queries):
                    categories = self.__classify_qstn(qg_pair=query_obj)
                    query_obj['actual_qstn_type'] = categories
                    filtered_queries[qi] = query_obj
                    
                query_arr[entity] = filtered_queries
                query_store["queries"] = query_arr

                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp)

                if entity in main_query_store["queries"]:
                    main_query_store["queries"][entity].extend(filtered_queries)
                else:
                    main_query_store["queries"][entity] = filtered_queries
            #main_query_store["queries"][entity] = filtered_queries
        
            with open(main_query_store_fp, 'w') as fp:
                json.dump(main_query_store, fp)

            os.remove(iquery_store_fp) # remove from intermediate storage after final dataset is constructed
                
        else:
            SystemExit('Chunk store not found!')

    def destroy(self):
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        os._exit(0)

if __name__ == "__main__":
    st = time()
    log_fp = f'logs/bu-answer-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)
    parser.add_argument('--no_of_entities', type = int, default = 20, required = False)

    args = parser.parse_args()

    qstn_classifier = QueryClassifier(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating answers for file: {args.filename}')
    qstn_classifier.set_filename(args.filename)
    qstn_classifier.classify_qstn(no_of_entities = args.no_of_entities)
    torch.cuda.empty_cache()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

    qstn_classifier.destroy()