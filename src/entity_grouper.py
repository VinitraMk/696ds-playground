from time import time
import sys
import argparse
import torch
import json
import re
import os
import gc

#custom imports
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import HF_CACHE_DIR, MODELS


os.environ['HF_HOME'] = HF_CACHE_DIR

class EntityGrouper:

    def __init__(self, model_index = 6):
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
            #self.llm = Together(api_key = cfg["togetherai_api_key"])
            self.model_folder = "llama"
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            #self.llm = AsyncGroq(api_key = cfg["groq_api_key"])
            self.model_folder = "llama"
        else:
            raise SystemExit('Invalid model index passed!')

    def group_entities(self, no_of_entities = 5, no_of_companies = 3):
        companies_to_anchor = ['NVDA', 'AMD', 'INTC', 'TSLA', 'F', 'GM']
        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}'
        
        entities_intersection = []
        entities_intersection_set = set([])
        for cp in companies_to_anchor:
            cp_filename = COMPANY_DICT[cp]['filename']
            cp_name = COMPANY_DICT[cp]['company_name']
            cp_entities_fp = os.path.join(chunk_store_fp, f'{cp_filename}_entities_info.json')
            company_entities_info = {}
            with open(cp_entities_fp, 'r') as fp:
                company_entities_info = json.load(fp)
            company_entities = list(company_entities_info.keys())
            if len(entities_intersection_set) == 0:
                entities_intersection_set = set(company_entities)
            else:
                entities_intersection_set = entities_intersection_set.intersection(set(company_entities))

            print('no common entities found so far: ', len(entities_intersection_set))

        entities_intersection = list(entities_intersection_set)
        print('no of common entities: ', len(entities_intersection_set))
        company_common_entities_info = {}
        optimum_range_count = 0
        for ek in entities_intersection:
            ek_count = 0
            for cp in companies_to_anchor:
                cp_filename = COMPANY_DICT[cp]['filename']
                cp_name = COMPANY_DICT[cp]['company_name']
                cp_entities_fp = os.path.join(chunk_store_fp, f'{cp_filename}_entities_info.json')
                company_entities_info = {}

                with open(cp_entities_fp, 'r') as fp:
                    company_entities_info = json.load(fp)

                if ek not in company_common_entities_info:
                    company_common_entities_info[ek] = {
                        
                    }
                else:
                    company_common_entities_info[ek][cp] = {
                        'count': company_entities_info[ek]['count'],
                        'chunk_indices': company_entities_info[ek]['chunk_indices']
                    }

                if company_entities_info[ek]['count'] >= 5 and company_entities_info[ek]['count'] <=20:
                    ek_count += 1
            if ek_count == 3:
                optimum_range_count += 1
        print('Entities in optimal range count (5, 20): ', optimum_range_count)
        with open('./company_common_entities_info.json', 'w') as fp:
            json.dump(company_common_entities_info, fp)
        
    def destroy(self):
        #del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        os._exit(0)



if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-entity-grouping-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    #torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type = int, default = 6, required = False)
    parser.add_argument('--no_of_entities', type = int, default = 5, required = False)
    parser.add_argument('--no_of_companies', type = int, default = 3, required = False)

    args = parser.parse_args()

    entity_grouper = EntityGrouper(model_index = args.model_index)
    entity_grouper.group_entities(no_of_entities = args.no_of_entities, no_of_companies = args.no_of_companies)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    entity_grouper.destroy()