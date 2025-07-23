import os
import torch
from vllm import LLM
import json
from time import time, sleep
import sys
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc
from google import genai
from together import Together
import random

from utils.string_utils import extract_json_array_by_key, is_valid_sentence, extract_json_text_by_key, extract_json_object_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer, execute_llama_task_api
from prompts.grounding_generation.grounding_prompts import GROUNDING_INSTRUCTION_PROMPT, GROUNDING_EVALUATION_PROMPT, GROUNDING_REFINEMENT_PROMPT


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
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
]


HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR

FILENAMES = [
    '10-K_AMD_20231230',
    '10-K_NVDA_20240128',
    '10-K_F_20231231',
    '10-K_GM_20231231',
    '10-K_INTC_20231230',
    '10-K_TSLA_20231231'
]

IGNORE_ENTITIES = ['Table of Contents', 'SEC', '10-K filings', 'SEC 10-K filings', 'SEC 10-K', 'SEC (Securities and Exchange Commission)', 'Notes', 'Item 1A', 'Part IV, Item 15', 'Item 601(b)(32)(ii)', 'Item 15', 'Item']

class GroundingsGenerator:

    def __init__(self, model_index = 6, prompt_batch_size = 3):
        self.model_name = MODELS[model_index]
        self.prompt_batch_size = prompt_batch_size

        with open("./config.json", "r") as fp:
            cfg = json.load(fp)
        self.device = torch.device("cuda")

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
        else:
            raise SystemExit('Invalid model index passed!')
        
    def __extract_and_clean_groundings(self, sentences):
        #factoid_list = extract_json_object_array_by_keys(text, ["factoid", "citation"])
        if len(sentences) > 0:
        #print('factoid list', factoid_list)
            clean_sentences = [s for s in sentences if is_valid_sentence(s, 200)]
        else:
            clean_sentences = []
        return clean_sentences

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

    def __generate_groundings(self, chunk_texts, entity, metadata):
        grounding_instruction_prompt = GROUNDING_INSTRUCTION_PROMPT

        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nText: {chunk['text']}\nEntity: {entity}\nMetadata: {metadata}" for chunk in chunk_texts]
        groundings_set = {}

        for gi, grounding_instruction_prompt in enumerate(grounding_instruction_prompts):
            groundings = []

            #initial groundings
            grounding_system_prompt = "You are a helpful assistant that given a chunk of text, an entity and some metadata about the text, returns groundings which are summarized statements related to the entity."
            gsummary = self.__get_output_from_llm(grounding_instruction_prompt, grounding_system_prompt)

            gjson_arr = extract_json_array_by_key(gsummary, "groundings")

            # grounding evaluation and refinement
            if gjson_arr != None and len(gjson_arr) > 0:
                gjson_arr_improved = gjson_arr
                gjson_arr_best = gjson_arr

                grounding_str = ",".join(gjson_arr_best)
                evaluation_system_prompt = "You are a helpful assistant that given a chunk of text, an entity, some metadata about the text and groundings which which are summarized statements related to the entity, returns an evaluation of the quality of the grounding."
                eval_prompt = GROUNDING_EVALUATION_PROMPT
                eval_prompt_text = eval_prompt.format(entity = entity,
                    chunk = chunk_texts[gi]['text'],
                    metadata = metadata,
                    groundings = grounding_str)

                esummary = self.__get_output_from_llm(eval_prompt_text, evaluation_system_prompt)
                evaluation_obj = extract_json_object_by_key(esummary, "evaluation")
                eval_best = evaluation_obj
                ts = evaluation_obj['entity_relevance'] + evaluation_obj['source_faithfulness'] + evaluation_obj['key_info_coverage'] + evaluation_obj['numeric_recall'] + evaluation_obj['non_redundancy']

                tsmax = ts
                no_of_attempts = 0
                while tsmax < 5 and no_of_attempts < 3:
                    
                    # grounding refinement
                    grounding_str = ",".join(gjson_arr_best)
                    eval_str = str(eval_best)
                    grounding_refinement_prompt = GROUNDING_REFINEMENT_PROMPT + f"\nText: {chunk_text}\nEntity: {entity}\nMetadata: {metadata}\nGroundings: {grounding_str}\nEvaluation: {eval_str}"
                    grounding_refinement_system_prompt = "You are a helpful assistant that given a chunk of text, an entity, some metadata about the text, groundings which are summarized statements related to the entity and their evaluation, returns an improved set of groundings"rounding_system_prompt = "You are a helpful assistant that given a chunk of text, an entity, some metadata about the text, groundings which are summarized statements related to the entity and their evaluation, returns an improved set of groundings"
                    grsummary = self.__get_output_from_llm(grounding_refinement_prompt, grounding_refinement_system_prompt)
                    gjson_arr_improved = extract_json_array_by_key(grsummary, "groundings")

                    # refined grounding evaluation
                    grounding_str = ",".join(gjson_arr_improved)
                    eval_prompt = GROUNDING_EVALUATION_PROMPT
                    eval_prompt_text = eval_prompt.format(entity = entity,
                        chunk = chunk_texts[gi]['text'],
                        metadata = metadata,
                        groundings = grounding_str)
                    esummary = self.__get_output_from_llm(eval_prompt_text, evaluation_system_prompt)
                    evaluation_obj = extract_json_object_by_key(esummary, "evaluation")
                    ts = evaluation_obj['entity_relevance'] + evaluation_obj['source_faithfulness'] + evaluation_obj['key_info_coverage'] + evaluation_obj['numeric_recall'] + evaluation_obj['non_redundancy']
                    if ts > tsmax:
                        tsmax = ts
                        gjson_arr_best = gjson_arr_improved
                        eval_best = evaluation_obj
                    no_of_attempts+=1
                    sleep(60)

                # best grounding cleanup
                if gjson_arr_best!= None and len(gjson_arr_best) > 0:
                    clean_g = self.__extract_and_clean_groundings(gjson_arr_best)
                    for gc in clean_g:
                        groundings.append({'text': gc, 'entity': entity})
            
            groundings_set[chunk_texts[gi]['chunk_index']] = groundings

        return groundings_set

    def __sample_entities(self, entities_info, count_range = (5, 15), k = 10):
        relevant_entities = {ek: entities_info[ek] for ek in entities_info.keys() if (ek not in IGNORE_ENTITIES) and (entities_info[ek]['count'] >= count_range[0]) and (entities_info[ek]['count'] <= count_range[1])}
        min_k = min(len(relevant_entities), k)
        sampled_entities = random.sample(relevant_entities.keys(), min_k)
        return sampled_entities

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]

    def generate_groundings(self, skip_entity_sampling = False):


        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
        entity_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_entities_info.json'
        sampled_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_sampled_entities.json'
        chunks_obj_fp = f'data/chunked_data/{self.filename}_chunked.json'

        if os.path.exists(chunk_store_fp) and os.path.exists(entity_store_fp):
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_store_arr = chunk_store["chunks"]

            with open(entity_store_fp, 'r') as fp:
                entity_store = json.load(fp)

            with open(chunks_obj_fp, 'r') as fp:
                chunks_obj = json.load(fp)
            chunk_arr = chunks_obj['chunks']

            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting grounding generation for each chunk\n')
            
            if not(skip_entity_sampling):
                sampled_entity_keys = self.__sample_entities(entities_info=entity_store, count_range=(5, 20), k = 20)
                print('\nSampled entities: ', sampled_entity_keys)
                with open(sampled_entities_fp, 'w') as fp:
                    json.dump({ "sampled_entities": sampled_entity_keys }, fp)
            else:
                with open(sampled_entities_fp, 'r') as fp:
                    sampled_entity_keys = json.load(fp)["sampled_entities"]
            if os.path.exists('../status.json'):
                with open('../status.json', 'r') as fp:
                    groundings_status_info = json.load(fp)
            else:
                groundings_status_info = {}
            for ek in sampled_entity_keys:
                if ek in groundings_status_info and groundings_status_info[ek] == "completed":
                    continue
                chunk_indices = entity_store[ek]["chunk_indices"]
                #chunk_entities = chunk_store_arr[ci]['entities']
                print(f'No of chunks for entity {ek}: ', len(chunk_indices))
                for cix in range(0, len(chunk_indices), self.prompt_batch_size):
                    cix_batch = chunk_indices[cix:cix+self.prompt_batch_size]
                    chunk_texts = [{ 'chunk_index': ci, 'text': chunk_arr[ci]} for ci in cix_batch]
                    entity_groundings = self.__generate_groundings(chunk_texts = chunk_texts, entity=ek, metadata=metadata)
                    for ci in cix_batch:
                        print(f'Grounding for {ek} in chunk {ci}: ', entity_groundings[ci])
                        if 'groundings' in chunk_store_arr[ci]:
                            chunk_store_arr[ci]['groundings'].extend(entity_groundings[ci])
                        else:
                            chunk_store_arr[ci]['groundings'] = entity_groundings[ci]
                chunk_store["chunks"] = chunk_store_arr
                with open(chunk_store_fp, 'w') as fp:
                    json.dump(chunk_store, fp)
            groundings_status_info = {}
            with open('../status.json', 'w') as fp:
                json.dump(groundings_status_info, fp)
        else:
            raise SystemExit('Chunk store not found!')

    def destroy(self):
        #del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        os._exit(0)

if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-groundings-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--skip_entity_sampling', type = bool, default = False, required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)

    args = parser.parse_args()

    ground_gen = GroundingsGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating groundings for file: {args.filename}')
    ground_gen.set_filename(args.filename)
    ground_gen.generate_groundings(skip_entity_sampling = args.skip_entity_sampling)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    ground_gen.destroy()