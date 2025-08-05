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
from groq import AsyncGroq

# custom imports
from utils.string_utils import extract_json_array_by_key, is_valid_sentence, extract_json_text_by_key, extract_json_object_by_key
from src.prompts.grounding_generation.grounding_prompts import GROUNDING_INSTRUCTION_PROMPT, GROUNDING_EVALUATION_PROMPT, GROUNDING_REFINEMENT_PROMPT
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS, NO_OF_TRIALS

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
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            self.llm = AsyncGroq(api_key = cfg["groq_api_key"])
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

    def __get_output_from_llm(self, instruction_prompts, system_prompt, json_schema = None, llm_config = None):
        summary = ""
        summary_stats = []
        if self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            prompt_tokens = self.tokenizer([get_prompt_token(instruction_prompts[0], system_prompt, self.tokenizer)], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            outputs = execute_llama_LLM_task(self.llm, prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            summary = outputs[0]
            if "Input for your task" in summary:
                ti = summary.index("Input for your task")
                summary = summary[ti:]
            summary = [summary]
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            summary = [execute_llama_task_api(self.llm, instruction_prompts[0], system_prompt)]
            print('generated response: ', summary)
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            summaries = execute_groq_task_api(self.llm, json_schema, instruction_prompts, system_prompt)
            summary = [robj['response'] for robj in summaries]
            for robj in summaries:
                new_obj = {k: v for k, v in robj.items() if k != "response"}
                summary_stats.append(new_obj)
            print('generated response: ', summaries[0])
        else:
            prompt_tokens = [get_prompt_token(instruction_prompts[0], system_prompt, self.tokenizer)]
            outputs = execute_LLM_tasks(self.llm, prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)
            summary = [outputs[0].outputs[0].text.strip()]
            print(f'generated response: ', summary)

        return summary, summary_stats

    def __generate_groundings(self, chunk_texts, entity, metadata):
        grounding_instruction_prompt = GROUNDING_INSTRUCTION_PROMPT

        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nText: {chunk['text']}\nEntity: {entity}\nMetadata: {metadata}" for chunk in chunk_texts]
        groundings_set = {}
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            groundings_json_schema = {
                "type": "json_object",
                "name": "groundings_generation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "groundings": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["groundings"],
                    "additionalProperties": False
                }
            }

            evaluation_json_schema = {
                "type": "json_object",
                "name": "groundings_generation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "evaluation": {
                            "type": "object",
                            "properties": {
                                "entity_relevance": {
                                    "type": "string"
                                },
                                "source_faithfulness": {
                                    "type": "string"
                                },
                                "key_info_coverage": {
                                    "type": "string"
                                },
                                "numeric_recall": {
                                    "type": "string"
                                },
                                "non_redundancy": {
                                    "type": "string"
                                },
                                "justification": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "required": ["evaluation"],
                    "additionalProperties": False
                }
            }
        else:
            evaluation_json_schema = None
            groundings_json_schema = None

        for gi, grounding_instruction_prompt in enumerate(grounding_instruction_prompts):
            groundings = []

            #initial groundings
            grounding_system_prompt = "You are a helpful assistant that given a chunk of text, an entity and some metadata about the text, returns groundings which are summarized statements related to the entity."
            gsummary, gstats = self.__get_output_from_llm([grounding_instruction_prompt], grounding_system_prompt, groundings_json_schema)

            gjson_arr = extract_json_array_by_key(gsummary[0], "groundings")

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

                esummary, _ = self.__get_output_from_llm([eval_prompt_text], evaluation_system_prompt, evaluation_json_schema)
                evaluation_obj = extract_json_object_by_key(esummary[0], "evaluation")
                eval_best = evaluation_obj
                gbest_stats = gstats
                ts = evaluation_obj['entity_relevance'] + evaluation_obj['source_faithfulness'] + evaluation_obj['key_info_coverage'] + evaluation_obj['numeric_recall'] + evaluation_obj['non_redundancy']
                #sleep(90)

                tsmax = ts
                no_of_attempts = 0
                while tsmax < 5 and no_of_attempts < NO_OF_TRIALS:
                    
                    # grounding refinement initialization
                    grounding_str = ",".join(gjson_arr_best)
                    eval_str = str(eval_best)
                    chunk_text = chunk_texts[gi]['text']
                    grounding_refinement_prompt = GROUNDING_REFINEMENT_PROMPT + f"\nText: {chunk_text}\nEntity: {entity}\nMetadata: {metadata}\nGroundings: {grounding_str}\nEvaluation: {eval_str}"
                    grounding_refinement_system_prompt = "You are a helpful assistant that given a chunk of text, an entity, some metadata about the text, groundings which are summarized statements related to the entity and their evaluation, returns an improved set of groundings"
                    grsummary, gstats = self.__get_output_from_llm([grounding_refinement_prompt], grounding_refinement_system_prompt, groundings_json_schema)
                    gjson_arr_improved = extract_json_array_by_key(grsummary[0], "groundings")
                    #sleep(90)

                    # refined grounding evaluation
                    if gjson_arr_improved != None and len(gjson_arr_improved) > 0:
                        grounding_str = ",".join(gjson_arr_improved)
                        eval_prompt = GROUNDING_EVALUATION_PROMPT
                        eval_prompt_text = eval_prompt.format(entity = entity,
                            chunk = chunk_texts[gi]['text'],
                            metadata = metadata,
                            groundings = grounding_str)
                        esummary, _ = self.__get_output_from_llm([eval_prompt_text], evaluation_system_prompt, evaluation_json_schema)
                        evaluation_obj = extract_json_object_by_key(esummary[0], "evaluation")
                        ts = evaluation_obj['entity_relevance'] + evaluation_obj['source_faithfulness'] + evaluation_obj['key_info_coverage'] + evaluation_obj['numeric_recall'] + evaluation_obj['non_redundancy']
                        if ts > tsmax:
                            tsmax = ts
                            gjson_arr_best = gjson_arr_improved
                            eval_best = evaluation_obj
                            gbest_stats = gstats
                        no_of_attempts+=1
                    #sleep(90)

                # best grounding cleanup
                if gjson_arr_best!= None and len(gjson_arr_best) > 0:
                    clean_g = self.__extract_and_clean_groundings(gjson_arr_best)
                    gop_avg = gbest_stats[0]['output_tokens'] / len(clean_g)
                    for gc in clean_g:
                        groundings.append({'text': gc, 'entity': entity, 'average_output_tokens': gop_avg})
            
            groundings_set[chunk_texts[gi]['chunk_index']] = groundings

        return groundings_set

    def __sample_entities(self, entities_info, count_range = (5, 15), k = 10):
        '''
        entities_to_ignore = [ek.lower() for ek in IGNORE_ENTITIES]
        relevant_entities = {ek: entities_info[ek] for ek in entities_info.keys() if (ek.lower() not in entities_to_ignore) and (entities_info[ek]['count'] >= count_range[0]) and (entities_info[ek]['count'] <= count_range[1])}
        min_k = min(len(relevant_entities), k)
        sampled_entities = random.sample(relevant_entities.keys(), min_k)
        return sampled_entities
        '''
        with open('./sampled_common_entities.json', 'r') as fp:
            sampled_common_entities = json.load(fp)["common_entities"]

        entities_to_ignore = [ek.lower() for ek in IGNORE_ENTITIES]
        relevant_entities = [ek for ek in sampled_common_entities if ek.lower() not in entities_to_ignore]
        sampled_entities = relevant_entities
        return sampled_entities

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]

    def generate_groundings(self, skip_entity_sampling = False, no_of_entities = 5):


        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
        entity_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_entities_info.json'
        sampled_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_sampled_entities.json'
        chunks_obj_fp = f'data/chunked_data/chunks/{self.filename}_chunked.json'
        groundings_status_fp = f'groundings_status.json'

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
                sampled_entity_keys = self.__sample_entities(entities_info=entity_store, count_range=(5, 20), k = no_of_entities)
                print('\nSampled entities: ', sampled_entity_keys)
                with open(sampled_entities_fp, 'w') as fp:
                    json.dump({ "sampled_entities": sampled_entity_keys }, fp)
            else:
                with open(sampled_entities_fp, 'r') as fp:
                    sampled_entity_keys = json.load(fp)["sampled_entities"]
            if os.path.exists(groundings_status_fp):
                with open(groundings_status_fp, 'r') as fp:
                    groundings_status_info = json.load(fp)
            else:
                groundings_status_info = {}
            entities_groundings_token_map = []
            for ek in sampled_entity_keys:
                if ek in groundings_status_info and groundings_status_info[ek] == "completed":
                    continue
                groundings_status_info[ek] = "ongoing"
                chunk_indices = entity_store[ek]["chunk_indices"]
                #chunk_entities = chunk_store_arr[ci]['entities']
                print(f'No of chunks for entity {ek}: ', len(chunk_indices))
                grounding_token_count = 0
                for cix in range(0, len(chunk_indices), self.prompt_batch_size):
                    cix_batch = chunk_indices[cix:cix+self.prompt_batch_size]
                    chunk_texts = [{ 'chunk_index': ci, 'text': chunk_arr[ci]} for ci in cix_batch]
                    entity_groundings = self.__generate_groundings(chunk_texts = chunk_texts, entity=ek, metadata=metadata)
                    for ci in cix_batch:
                        print(f'Grounding for {ek} in chunk {ci}: ', entity_groundings[ci])
                        if 'groundings' in chunk_store_arr[ci] and 'groundings_token_count' in chunk_store_arr[ci]:
                            chunk_store_arr[ci]['groundings'].extend(entity_groundings[ci])
                            chunk_store_arr[ci]['groundings_token_count'] += sum(map(lambda x: x['average_output_tokens'], entity_groundings[ci]))
                        else:
                            chunk_store_arr[ci]['groundings'] = entity_groundings[ci]
                            chunk_store_arr[ci]['groundings_token_count'] = sum(map(lambda x: x['average_output_tokens'], entity_groundings[ci]))
                        grounding_token_count += chunk_store_arr[ci]['groundings_token_count']
                entities_groundings_token_map.append({'entity': ek, 'groundings_token_count': grounding_token_count})
                groundings_status_info[ek] = "completed"
                chunk_store["chunks"] = chunk_store_arr
                with open(chunk_store_fp, 'w') as fp:
                    json.dump(chunk_store, fp)
                with open(groundings_status_fp, 'w') as fp:
                    json.dump(groundings_status_info, fp)
            groundings_status_info = {}
            entities_groundings_token_map = sorted(entities_groundings_token_map, key = lambda x: x['groundings_token_count'])
            with open('./sampled_common_entities.json', 'r') as fp:
                sampled_entities_obj = json.load(fp)
            sampled_entities_obj['optimal_entities'][self.filename] = entities_groundings_token_map
            with open('./sampled_common_entities.json', 'w') as fp:
                json.dump(sampled_entities_obj, fp)
            with open(groundings_status_fp, 'w') as fp:
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
    #torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--skip_entity_sampling', type = bool, default = False, required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False) # prompt batch size to be set as 1 for this always
    parser.add_argument('--no_of_entities', type = int, default = 5, required = False)

    args = parser.parse_args()

    ground_gen = GroundingsGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating groundings for file: {args.filename}')
    ground_gen.set_filename(args.filename)
    ground_gen.generate_groundings(skip_entity_sampling = args.skip_entity_sampling, no_of_entities = args.no_of_entities)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    ground_gen.destroy()