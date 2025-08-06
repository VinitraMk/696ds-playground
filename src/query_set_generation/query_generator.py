import os
import torch
import numpy as np
import json
from time import time, sleep
import sys
import argparse
import random
from typing import List, Any, Dict

# custom imports
from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from src.prompts.query_set_generation.summarization_question_prompt import SUMM_QSTN_INSTRUCTION_PROMPT
from src.prompts.query_set_generation.temporal_analysis_question_prompt import TMP_QSTN_INSTRUCTION_PROMPT
from src.prompts.query_set_generation.entity_interaction_analysis_question_prompt import ETINT_QSTN_INSTRUCTION_PROMPT
from src.prompts.query_set_generation.event_interaction_analysis_question_prompt import EVTINT_QSTN_INSTRUCTION_PROMPT
from src.prompts.query_set_generation.numerical_analysis_question_prompt import NUM_QSTN_INSTRUCTION_PROMPT
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS, MAX_GROUNDINGS_TO_SAMPLE, MIN_GROUNDINGS_NEEDED_FOR_GENERATION, QUERY_INDEX
from src.base.generator import Generator

QUERY_JSON_SCHEMA = {
    "type": "json_object",
    "name": "query_generation",
    "schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["queries"],
        "additionalProperties": False
    }
}
QUERY_SYSTEM_PROMPT = "You are a helpful assistant, that given a list of groundings, generates meaningful and complex questions from it."

class QueryGenerator(Generator):

    def __init__(self, model_index:int = 6, prompt_batch_size: int = 1,
        query_type:str = 'summarization', query_hop_span: str = 'multi_doc'):
        super().__init__(model_index = model_index, prompt_batch_size = prompt_batch_size)
        self.query_type = query_type
        self.query_hop_span = query_hop_span

    def __generate_queries_in_single_prompt(self, groundings_doc_text, metadata, entity):

        query_instruction_prompts = []
        if self.query_type == 'all':
            for qt in QUERY_INDEX.keys():
                if QUERY_INDEX[qt] == 'entity_interaction_analysis':
                    qstn_instruction_prompt = ETINT_QSTN_INSTRUCTION_PROMPT
                elif QUERY_INDEX[qt] == 'temporal_analysis':
                    qstn_instruction_prompt = TMP_QSTN_INSTRUCTION_PROMPT
                elif QUERY_INDEX[qt] == 'event_interaction_analysis':
                    qstn_instruction_prompt = EVTINT_QSTN_INSTRUCTION_PROMPT
                elif QUERY_INDEX[qt] == 'numerical_analysis':
                    qstn_instruction_prompt = NUM_QSTN_INSTRUCTION_PROMPT
                else:
                    qstn_instruction_prompt = SUMM_QSTN_INSTRUCTION_PROMPT

                qstn_instruction_prompt = qstn_instruction_prompt + f"\nMetadata: {metadata}\nGroundings: {groundings_doc_text}\nEntity: {entity}"
                query_instruction_prompts.append(qstn_instruction_prompt)
        else:
            if self.query_type == 'entity_interaction_analysis':
                qstn_instruction_prompt = ETINT_QSTN_INSTRUCTION_PROMPT
            elif self.query_type == 'temporal_analysis':
                qstn_instruction_prompt = TMP_QSTN_INSTRUCTION_PROMPT
            elif self.query_type == 'event_interaction_analysis':
                qstn_instruction_prompt = EVTINT_QSTN_INSTRUCTION_PROMPT
            elif self.query_type == 'numerical_analysis':
                qstn_instruction_prompt = NUM_QSTN_INSTRUCTION_PROMPT
            else:
                qstn_instruction_prompt = SUMM_QSTN_INSTRUCTION_PROMPT
            qstn_instruction_prompt = qstn_instruction_prompt + f"\nMetadata: {metadata}\nGroundings: {groundings_doc_text}\nEntity: {entity}"
            query_instruction_prompts = [qstn_instruction_prompt]

        qstn_system_prompt = QUERY_SYSTEM_PROMPT
        query_strs = []
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            query_json_schema = QUERY_JSON_SCHEMA
            qsummaries, _ = self.get_output_from_llm(query_instruction_prompts, qstn_system_prompt, query_json_schema)
        else:
            qsummaries, _ = self.get_output_from_llm(query_instruction_prompts, qstn_system_prompt, None)

        for qsummary in qsummaries:
            qjson = extract_json_array_by_key(qsummary, "queries")
            if qjson != None and len(qjson) > 0:
                query_strs.extend(qjson)
        
        return query_strs
        
    def __merge_all_groundings(self, companies):
        
        all_groundings = []
        for cp in companies:
            fn = COMPANY_DICT[cp]["filename"]
            cp_name = COMPANY_DICT[cp]['company_name']
            chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{cp}/{fn}_chunk_store.json'
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_arr = chunk_store["chunks"]

            for ci, chunk in enumerate(chunk_arr):
                if "groundings" in chunk and len(chunk["groundings"]) > 0:
                    #filtered_groundings = [gobj for gobj in chunk["groundings"] if gobj['entity'].lower() == entity.lower()]
                    #for gi, gobj in enumerate(filtered_groundings):
                        #filtered_groundings[gi] = gobj | { 'company_code': cp, 'company_name': cp_name, 'chunk_index': ci }
                    #all_groundings.append({'company_name': cp_name, 'chunk_index': ci, 'groundings': filtered_groundings})
                    all_groundings.extend(chunk["groundings"])
        return all_groundings

    def __get_questions(self,
        filtered_groundings: List[Any], all_resp: Dict[str, Any],
        entity: str, docs_considered = List[str],
        no_of_qstns: int = 1):

        while len(all_resp[entity]) < (no_of_qstns * 5):
            '''
            for fbi,i in enumerate(range(0, len(filtered_groundings), MAX_GROUNDINGS_TO_SAMPLE)):
                groundings_subarr = filtered_groundings[i:i+MAX_GROUNDINGS_TO_SAMPLE]
                if len(groundings_subarr) < MIN_GROUNDINGS_NEEDED_FOR_GENERATION:
                    max_len = max(MAX_GROUNDINGS_TO_SAMPLE, len(filtered_groundings))
                    groundings_subarr = filtered_groundings[-max_len:]
            '''
            groundings_subarr = filtered_groundings    
            chunks_reduced = [(gobj['doc_code'], gobj['chunk_index']) for gobj in groundings_subarr]
            chunks_used = list(set(chunks_reduced))
            chunks_used = [{ 'doc_code': cobj[0], 'chunk_index': cobj[1]} for cobj in chunks_used]
            #groundings_str = "[" + ",\n".join("{'text':" + f"{item['text']}" + ", 'company_name':" + f"{item['company_name']}" + "}" for item in groundings_subarr) + "]"
            #print('groundings subarr', groundings_subarr)
            groundings_str = json.dumps(
                [{"text": item["text"], "company_name": COMPANY_DICT[item["doc_code"]]["company_name"]} for item in groundings_subarr],
                indent=2
            )
            print(f'\nRunning query  generation for entity: ', entity)
            metadata = f'Input source: SEC 10-K Filings | Companies addressed in the groundings: {",".join(list(map(lambda x: COMPANY_DICT[x]["company_name"], docs_considered)))}'
            query_strs = self.__generate_queries_in_single_prompt(groundings_str, metadata, entity)
            all_resp[entity].extend([{'query': query_str, 'query_hop_span': self.query_hop_span, 'docs_considered': docs_considered, 'groundings': groundings_subarr, 'chunks_used': chunks_used, 'intended_query_type': [QUERY_INDEX[qi]] } for qi, query_str in enumerate(query_strs)])
            print(f'No of queries formed using entity {entity}: ', len(all_resp[entity]))
            
        return all_resp

    def generate_query(self, no_of_qstns:int = 5, no_of_entities:int = 20):

        #all_resp = []
        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filecode}/{self.filename}_chunk_store.json'
        sampled_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filecode}/{self.filename}_sampled_entities.json'
        sampled_doc_groups_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/sampled_doc_groups.json'

        if os.path.exists(chunk_store_fp) and ((self.query_hop_span == 'single_doc' and os.path.exists(sampled_entities_fp)) or (self.query_hop_span == 'multi_doc' and os.path.exists(sampled_doc_groups_fp))):
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_arr = chunk_store["chunks"]

            # get sampled entities / doc groups
            if self.query_hop_span == 'single_doc':
                with open(sampled_entities_fp, 'r') as fp:
                    sampled_entities = json.load(fp)['sampled_entities']
                iquery_json_path = f'./intermediate_data/query_sets/{self.model_folder}/single_doc/{self.filecode}/{self.filename}_generated_queries.json'
            else:
                iquery_json_path = f'./intermediate_data/query_sets/{self.model_folder}/multi_doc/multi_doc_generated_queries.json'
                #sampled_entities = ["Ai", "Intangible Assets", "Data Center"]
                with open(sampled_doc_groups_fp, 'r') as fp:
                    sampled_doc_groups_info = json.load(fp)

                sampled_entities = list(sampled_doc_groups_info.keys())
                #relevant_companies = ['NVDA', 'AMD', 'INTC']
                relevant_companies = set([])
                for sek in sampled_entities:
                    doc_groups = sampled_doc_groups_info[sek]
                    for dcg in doc_groups:
                        for doc in dcg:
                            relevant_companies.add(doc)
                relevant_companies = list(relevant_companies)

            all_groundings = []
            if self.query_hop_span == 'single_doc':
                for ci, chunk in enumerate(chunk_arr):
                    if "groundings" in chunk and len(chunk["groundings"]) > 0:
                        all_groundings.append({'chunk_index': ci, 'groundings': chunk["groundings"]})
            else:
                all_groundings = self.__merge_all_groundings(relevant_companies)

            print('length of the entire groundings array: ', len(all_groundings))

            all_resp = {}
            print('\nStarting query generation for batch of groundings\n')
            total_q = 0

            for ei in range(no_of_entities):
                entity = sampled_entities[ei]
                all_resp[entity] = []
                filtered_groundings = []

                if self.query_hop_span == 'single_doc':
                    for gobj in all_groundings:
                        ci = gobj['chunk_index']
                        rel_groundings = [{'chunk_index': ci, 'entity': entity, 'text': gr['text'] } for gr in gobj['groundings'] if gr['entity'] == entity]
                        filtered_groundings.extend(rel_groundings)
                    
                    print(f'No of groundings under entity {entity}: ', len(filtered_groundings))
                    
                    all_resp = self.__get_questions(filtered_groundings = filtered_groundings, all_resp = all_resp,
                        entity = entity, docs_considered = [self.filecode],
                        no_of_qstns = no_of_qstns)
                else:
                    doc_groups = sampled_doc_groups_info[entity]
                    filtered_groundings = []
                    print('doc groups', doc_groups)
                    #print('\nall groundings: ', len(all_groundings), all_groundings)
                    for dcg in doc_groups:
                        filtered_groundings.extend([gobj for gobj in all_groundings if gobj['entity'] == entity and gobj['doc_code'] in dcg])
                        #print('filtered groundings: ', filtered_groundings)
                        print(f'No of groundings under entity {entity}: ', len(filtered_groundings))
                        if len(filtered_groundings) > 0:
                            all_resp = self.__get_questions(filtered_groundings = filtered_groundings, all_resp = all_resp,
                            entity = entity, docs_considered = dcg,
                            no_of_qstns = no_of_qstns)
                    total_q += len(all_resp[entity])
                
            if all_resp != {}:
                print('\nTotal no of questions generated: ', total_q)
                #iquery_json_path = f'./intermediate_data/query_sets/{self.model_folder}/{self.filename}_generated_queries.json'
                queries = { 'queries': {} }
                queries["queries"] = all_resp

                with open(iquery_json_path, 'w') as fp:
                    json.dump(queries, fp)
        else:
            raise SystemExit('Chunk store not found!')

if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-query-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 5, required = False)
    parser.add_argument('--filecode', type = str, default = 'NVDA', required = False)
    parser.add_argument('--no_of_entities', type = int, default = 20, required = False)
    parser.add_argument('--query_type', type = str, default = 'summarization', required = False)
    parser.add_argument('--query_hop_span', type = str, default = 'multi_doc', required = False)

    args = parser.parse_args()

    query_gen = QueryGenerator(model_index = args.model_index, query_type = args.query_type, query_hop_span = 'single_doc')
    print(f'\n\nGenerating queries for file: {args.filecode}')
    query_gen.set_filename(args.filecode)
    query_gen.generate_query(no_of_qstns = args.no_of_qstns, no_of_entities = args.no_of_entities)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
