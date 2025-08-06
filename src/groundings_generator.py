import os
import torch
import json
from time import time, sleep
import sys
import argparse
import random

# custom imports
from utils.string_utils import extract_json_array_by_key, is_valid_sentence, extract_json_text_by_key, extract_json_object_by_key
from src.prompts.grounding_generation.grounding_prompts import GROUNDING_INSTRUCTION_PROMPT, GROUNDING_EVALUATION_PROMPT, GROUNDING_REFINEMENT_PROMPT, GROUNDING_SYSTEM_PROMPT, GROUNDING_REFINEMENT_SYSTEM_PROMPT, GROUNDING_EVAL_SYSTEM_PROMPT, GROUNDING_JSON_SCHEMA, GROUNDING_EVAL_JSON_SCHEMA
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS, NO_OF_TRIALS
from src.base.generator import Generator

class GroundingsGenerator(Generator):

    def __init__(self, model_index:int = 6, prompt_batch_size:int = 1):
        super().__init__(model_index = model_index, prompt_batch_size = prompt_batch_size)

    def __extract_and_clean_groundings(self, sentences):
        if len(sentences) > 0:
            clean_sentences = [s for s in sentences if is_valid_sentence(s, 400)]
        else:
            clean_sentences = []
        return clean_sentences

    def __generate_groundings(self, chunk_texts, entity, metadata):
        grounding_instruction_prompt = GROUNDING_INSTRUCTION_PROMPT

        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nText: {chunk['text']}\nEntity: {entity}\nMetadata: {metadata}" for chunk in chunk_texts]
        groundings_set = {}
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            groundings_json_schema = GROUNDING_JSON_SCHEMA
            evaluation_json_schema = GROUNDING_EVAL_JSON_SCHEMA
        else:
            evaluation_json_schema = None
            groundings_json_schema = None

        eval_prompt = GROUNDING_EVALUATION_PROMPT
        grounding_system_prompt = GROUNDING_SYSTEM_PROMPT
        evaluation_system_prompt = GROUNDING_EVAL_SYSTEM_PROMPT
        grounding_refinement_system_prompt = GROUNDING_REFINEMENT_SYSTEM_PROMPT

        for gi, grounding_instruction_prompt in enumerate(grounding_instruction_prompts):
            groundings = []
            grounding_versions = {}

            #initial groundings
            gsummary, gstats = self.get_output_from_llm([grounding_instruction_prompt], grounding_system_prompt, groundings_json_schema)
            gjson_arr = extract_json_array_by_key(gsummary[0], "groundings")

            # grounding evaluation and refinement
            if gjson_arr != None and len(gjson_arr) > 0:
                gjson_arr_improved = gjson_arr
                gjson_arr_best = gjson_arr

                grounding_str = ",".join(gjson_arr_best)
                eval_prompt_text = eval_prompt.format(entity = entity,
                    chunk = chunk_texts[gi]['text'],
                    metadata = metadata,
                    groundings = grounding_str)

                esummary, _ = self.get_output_from_llm([eval_prompt_text], evaluation_system_prompt, evaluation_json_schema)
                evaluation_obj = extract_json_object_by_key(esummary[0], "evaluation")
                eval_best = evaluation_obj
                gbest_stats = gstats
                ts = evaluation_obj['entity_relevance'] + evaluation_obj['source_faithfulness'] + evaluation_obj['key_info_coverage'] + evaluation_obj['numeric_recall'] + evaluation_obj['non_redundancy']
                #sleep(90)

                tsmax = ts
                no_of_attempts = 0
                grv = len(grounding_versions.keys())
                grounding_versions[grv] = gjson_arr

                while tsmax < 5 and no_of_attempts < NO_OF_TRIALS:
                    
                    # grounding refinement initialization
                    grounding_str = ",".join(gjson_arr_best)
                    eval_str = str(eval_best)
                    chunk_text = chunk_texts[gi]['text']
                    grounding_refinement_prompt = GROUNDING_REFINEMENT_PROMPT + f"\nText: {chunk_text}\nEntity: {entity}\nMetadata: {metadata}\nGroundings: {grounding_str}\nEvaluation: {eval_str}"
                    grsummary, gstats = self.get_output_from_llm([grounding_refinement_prompt], grounding_refinement_system_prompt, groundings_json_schema)
                    gjson_arr_improved = extract_json_array_by_key(grsummary[0], "groundings")
                    #sleep(90)

                    # refined grounding evaluation
                    if gjson_arr_improved != None and len(gjson_arr_improved) > 0:
                        grv = len(grounding_versions.keys())
                        grounding_versions[grv] = gjson_arr_improved
                        grounding_str = ",".join(gjson_arr_improved)
                        eval_prompt_text = eval_prompt.format(entity = entity,
                            chunk = chunk_texts[gi]['text'],
                            metadata = metadata,
                            groundings = grounding_str)
                        esummary, _ = self.get_output_from_llm([eval_prompt_text], evaluation_system_prompt, evaluation_json_schema)
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
                    #clean_g = self.__extract_and_clean_groundings(gjson_arr_best)
                    gop_avg = gbest_stats[0]['output_tokens'] / len(gjson_arr_best)
                    for gc in gjson_arr_best:
                        groundings.append({'doc_code': self.filecode, 'text': gc, 'entity': entity, 'average_output_tokens': gop_avg})
            
            groundings_set[chunk_texts[gi]['chunk_index']] = {
                'best_groundings': groundings,
                'all_versions': grounding_versions
            }

        return groundings_set

    def __sample_entities(self, entities_info, count_range = (5, 15), k = 10):
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
        '''

    def generate_groundings(self, skip_entity_sampling = False,
        use_bucket_entities = False,
        no_of_entities = 5,
        bucket_size = 2):


        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filecode}/{self.filename}_chunk_store.json'
        entity_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filecode}/{self.filename}_entities_info.json'
        sampled_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filecode}/{self.filename}_sampled_entities.json'
        bucket_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/all_doc_entity_stats.json'
        bucket_entities_info_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/all_doc_entity_grounding_stats.json'
        chunks_obj_fp = f'data/chunked_data/chunks/{self.filename}_chunked.json'

        if os.path.exists(chunk_store_fp) and os.path.exists(chunks_obj_fp) and ((use_bucket_entities and os.path.exists(bucket_entities_fp)) or (not(skip_entity_sampling) and os.path.exists(entity_store_fp)) or (skip_entity_sampling and os.path.exists(sampled_entities_fp))):
            
            # get chunks from chunk store
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_store_arr = chunk_store["chunks"]

            # get raw chunks
            with open(chunks_obj_fp, 'r') as fp:
                chunks_obj = json.load(fp)
            chunk_arr = chunks_obj['chunks']

            # get entities
            if use_bucket_entities:
                with open(bucket_entities_fp, 'r') as fp:
                    entity_store = json.load(fp)[f"{bucket_size}"]
                with open(entity_store_fp, 'r') as fp:
                    doc_entity_store = json.load(fp)
                sampled_entity_keys = list(entity_store.keys())
                sampled_entity_keys = [ek for ek in sampled_entity_keys if ek in doc_entity_store.keys()]
            elif skip_entity_sampling:
                with open(sampled_entities_fp, 'r') as fp:
                    sampled_entity_keys = json.load(fp)["sampled_entities"]
            else:
                with open(entity_store_fp, 'r') as fp:
                    entity_store = json.load(fp)
                sampled_entity_keys = self.__sample_entities(entities_info=entity_store, count_range=(5, 20), k = no_of_entities)
                print('\nSampled entities: ', sampled_entity_keys)
                with open(sampled_entities_fp, 'w') as fp:
                    json.dump({ "sampled_entities": sampled_entity_keys }, fp)

            
            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting grounding generation for each chunk\n')
            
            groundings_status_info = self.get_script_status()
            doc_ent_groundings_token_map = []
            for ek in sampled_entity_keys:
                if ek in groundings_status_info and groundings_status_info[ek] == "completed":
                    continue
                self.update_script_status(groundings_status_info, ek, "ongoing")
                if use_bucket_entities:
                    doc_groups = entity_store[ek]
                    found = False
                    for dg in doc_groups:
                        for dobj in dg:
                            if dobj['doc'] == self.filecode:
                                chunk_indices = dobj["chunk_indices"]
                                found = True
                                break
                        if found:
                            break
                else:
                    chunk_indices = entity_store[ek]["chunk_indices"]
                print(f'No of chunks for entity {ek}: ', len(chunk_indices))

                grounding_token_count = 0
                for cix in range(0, len(chunk_indices), self.prompt_batch_size):
                    cix_batch = chunk_indices[cix:cix+self.prompt_batch_size]
                    chunk_texts = [{ 'chunk_index': ci, 'text': chunk_arr[ci]} for ci in cix_batch]
                    entity_groundings = self.__generate_groundings(chunk_texts = chunk_texts, entity=ek, metadata=metadata)
                    for ci in cix_batch:
                        print(f'Grounding for {ek} in chunk {ci}: ', entity_groundings[ci]['best_groundings'])
                        if 'groundings' in chunk_store_arr[ci] and 'groundings_token_count' in chunk_store_arr[ci]:
                            chunk_store_arr[ci]['groundings'].extend(entity_groundings[ci]['best_groundings'])
                            chunk_store_arr[ci]['groundings_versions'].append(entity_groundings[ci]['all_versions'])
                            chunk_store_arr[ci]['groundings_token_count'] += sum(map(lambda x: x['average_output_tokens'], entity_groundings[ci]['best_groundings']))
                        else:
                            chunk_store_arr[ci]['groundings'] = entity_groundings[ci]['best_groundings']
                            chunk_store_arr[ci]['groundings_versions'] = [entity_groundings[ci]['all_versions']]
                            chunk_store_arr[ci]['groundings_token_count'] = sum(map(lambda x: x['average_output_tokens'], entity_groundings[ci]['best_groundings']))
                        grounding_token_count += chunk_store_arr[ci]['groundings_token_count']

                doc_ent_groundings_token_map.append({'entity': ek, 'groundings_token_count': grounding_token_count})

                #update results and status
                self.update_script_status(groundings_status_info, ek, "completed")
                chunk_store["chunks"] = chunk_store_arr
                with open(chunk_store_fp, 'w') as fp:
                    json.dump(chunk_store, fp)

            doc_ent_groundings_token_map = sorted(doc_ent_groundings_token_map, key = lambda x: x['groundings_token_count'])

            if os.path.exists(bucket_entities_info_fp):
                with open(bucket_entities_info_fp, 'r') as fp:
                    bucket_entities_info = json.load(fp)
            else:
                bucket_entities_info = {}

            if self.filecode not in bucket_entities_info:
                bucket_entities_info[self.filecode] = {}

            bucket_entities_info[self.filecode] = doc_ent_groundings_token_map
            with open(bucket_entities_info_fp, 'w') as fp:
                json.dump(bucket_entities_info, fp)
            self.reset_script_status()
        else:
            raise SystemExit('Chunk store not found!')

if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-groundings-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filecode', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--skip_entity_sampling', type = bool, default = False, required = True)
    parser.add_argument('--use_bucket_entities', type = bool, default = False, required = True)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False) # prompt batch size to be set as 1 for this always
    parser.add_argument('--no_of_entities', type = int, default = 5, required = False)
    parser.add_argument('--bucket_size', type = int, default = 2, required = False)

    args = parser.parse_args()

    ground_gen = GroundingsGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating groundings for filecode: {args.filecode}')
    ground_gen.set_filename(filecode = args.filecode)
    ground_gen.generate_groundings(skip_entity_sampling = args.skip_entity_sampling,
        use_bucket_entities = args.use_bucket_entities,
        no_of_entities = args.no_of_entities,
        bucket_size = args.bucket_size)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()