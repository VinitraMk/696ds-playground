import torch
import os
import sys
import json
from time import time
from vllm import LLM
import argparse
import multiprocessing
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from together import Together
import matplotlib.pyplot as plt

#custom imports
from utils.string_utils import extract_json_array_by_key, is_valid_sentence, extract_json_object_array_by_keys, extract_json_text_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, get_tokenizer, execute_llama_task_api
from prompts.entity_generation.entity_prompt import ENTITY_INSTRUCTION_PROMPT

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
    "gemini-2.0-flash",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR

class EntityGen:

    def __init__(self, filename, model_index = 0):
        self.filename = filename
        self.model_index = model_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f'Device enabled: {self.device}')

        with open("./config.json", "r") as fp:
            cfg = json.load(fp)

        # Load the model using vLLM
        self.model_name = MODELS[self.model_index]
        print('Model used: ', self.model_name)
        self.tokenizer = get_tokenizer(model_name = self.model_name)

        if "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization = 0.95,
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
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
            print('Invalid model name passed!')
            SystemExit()


    def __extract_and_clean_factoids(self, text):
        #factoid_list = extract_json_object_array_by_keys(text, ["factoid", "citation"])
        factoid_list = extract_json_array_by_key(text, "factoids")
        if (factoid_list) and (len(factoid_list) > 0):
        #print('factoid list', factoid_list)
            clean_factoids_citations = [s for s in factoid_list if is_valid_sentence(s)]
        else:
            clean_factoids_citations = []
        return clean_factoids_citations

    def __get_output_from_llm(self, instruction_prompt, system_prompt, llm_config = None):
        summary = ""
        if self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            prompt_tokens = self.tokenizer([get_prompt_token(instruction_prompt, system_prompt, self.tokenizer)], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            outputs = execute_llama_LLM_task(self.llm, prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            summary = outputs[0]
            if "Input for your task" in summary:
                ti = summary.index("Input for your task")
                summary = summary[ti:]
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            summary = execute_llama_task_api(self.llm, instruction_prompt, system_prompt)
            print('generated response: ', summary)
        else:
            prompt_tokens = [get_prompt_token(instruction_prompt, system_prompt, self.tokenizer)]
            outputs = execute_LLM_tasks(self.llm, prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)
            summary = outputs[0].outputs[0].text.strip()
            print(f'generated response: ', summary)

        return summary

    def __generate_entities_from_chunk(self, chunks):

        entity_instruction_prompt = ENTITY_INSTRUCTION_PROMPT
        
        entity_system_prompt = "You are a helpful assistant, that given a chunk of text, generates entites addressed in the text."
        chunk_entities = []
        entity_info = {}
        entity_chunk_info = {}

        for ci,chunk in enumerate(chunks):
            entity_prompt_text = entity_instruction_prompt + f"\nChunk: {chunk}"
            esummary = self.__get_output_from_llm(entity_prompt_text, entity_system_prompt)
            ejson = extract_json_array_by_key(esummary, "entities")
            if ejson != None and len(ejson) > 0:
                print(f'Entities from chunk {ci}:', ejson)
                for en in ejson:
                    if en in entity_info:
                        entity_info[en] = entity_info[en] + 1
                        entity_chunk_info[en].append(ci)
                    else:
                        entity_info[en] = 1
                        entity_chunk_info[en] = [ci]
                chunk_entities.append(ejson)
            else:
                chunk_entities.append([])
        count_vals = list(entity_info.values())
        count_keys = list(entity_info.keys())
        entity_vks = sorted(list(zip(count_vals, count_keys)), reverse=True)
        sorted_entity_info = {}
        for el in entity_vks:
            if entity_info[el[1]] == el[0]:
                sorted_entity_info[el[1]] = { 'count': entity_info[el[1]], 'chunk_indices': entity_chunk_info[el[1]] }

        return chunk_entities, sorted_entity_info
    
    def generate_entities(self):

        chunk_fn = f'{self.filename}_chunked'
        chunk_fp = f'data/chunked_data/{chunk_fn}.json'
        scored_chunk_fp = f'data/chunked_data/scored_chunks/{chunk_fn}.json'
        with open(chunk_fp, 'r') as fp:
            self.all_chunks = json.load(fp)["chunks"]
        with open(scored_chunk_fp, 'r') as fp:
            self.scored_chunks = json.load(fp)
        self.chunks = [{ 'chunk_index': ci, 'text': self.all_chunks[ci] } for ci in range(len(self.scored_chunks))]
        print('Filtered chunks: ', len(self.chunks))

        file_chunkstore_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
        if os.path.exists(file_chunkstore_fp):
            with open(file_chunkstore_fp, 'r+') as fp:
                chunks_obj = json.load(fp)
        else:
            chunks_obj = { "chunks": [] }

        all_resp = []
        chunk_entities, entity_info = self.__generate_entities_from_chunk(self.chunks)

        entity_count_values = [eiob['count'] for eiob in list(entity_info.values())]
        entity_values = list(entity_info.keys())
        plt.bar(entity_values, entity_count_values)
        plt.savefig(f'./figures/entity_plots/{self.filename}_entity_dist.png')

        # save entities info
        with open(f'./data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_entities_info.json', 'w') as fp:
            json.dump(entity_info, fp)

        #print(chunk_factoids)
        all_resp = []
        for i in range(len(self.all_chunks)):
            chunk_resp = {
                'chunk_index': i,
                'entities': chunk_entities[i],
            }
            all_resp.append(chunk_resp)
        chunks_obj["chunks"] = all_resp
        if os.path.exists(file_chunkstore_fp):
            with open(file_chunkstore_fp, 'w') as fp:
                json.dump(chunks_obj, fp)
        else:
            fp = open(file_chunkstore_fp, 'x')
            json.dump(chunks_obj, fp)


if __name__ == "__main__":
    
    st = time()
    log_fp = f'logs/bu-entity-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type=str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    entity_gen = EntityGen(filename = args.filename, model_index = args.model_index)
    entity_gen.generate_entities()
    

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
