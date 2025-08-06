import os
import torch
import json
from time import time, sleep
import sys
import argparse
from typing import List

# custom imports
from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from src.prompts.query_set_generation.question_classifier import QUERY_CLASSIFICATION_INSTRUCTION_PROMPT, QUERY_CLASSIFICATION_SYSTEM_PROMPT, QUERY_CLASSIFICATION_JSON_SCHEMA
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS
from src.base.generator import Generator

class QueryClassifier(Generator):

    def __init__(self, model_index = 6, prompt_batch_size = 1, query_hop_span: str = 'multi_doc'):
        super().__init__(model_index = model_index, prompt_batch_size = prompt_batch_size)
        self.query_hop_span = query_hop_span
        
    def __classify_query(self, query_strs: List[str]):

        categories = [[]] * len(query_strs)
        query_classification_instruction_prompts = [QUERY_CLASSIFICATION_INSTRUCTION_PROMPT + f"\nQuery: {qstr}" for qstr in query_strs]
        query_classification_system_prompt = QUERY_CLASSIFICATION_SYSTEM_PROMPT
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            qclass_json_schema = QUERY_CLASSIFICATION_JSON_SCHEMA
        else:
            qclass_json_schema = None
        
        qclass_summaries, _ = self.get_output_from_llm(query_classification_instruction_prompts, query_classification_system_prompt, qclass_json_schema)
        for qci, qclass_summary in enumerate(qclass_summaries):
            qclass_json = extract_json_array_by_key(qclass_summary, "categories")
            if qclass_json != None and len(qclass_json) > 0:
                #categories = list(map(lambda q: QUERY_TYPE_MAP[q], qclass_json))
                categories[qci] = qclass_json
            else:
                categories[qci] = []
            
        return categories

    def classify_query(self, no_of_entities = 20):
        
        if self.query_hop_span == "multi_doc":
            iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/multi_doc/multi_doc_generated_queries.json'
            main_query_store_fp = f'data/queries/{self.model_folder}/multi_doc/multi_doc_generated_queries.json'
        else:
            iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/single_doc/{self.filecode}/{self.filename}_generated_queries.json'
            main_query_store_fp = f'data/queries/{self.model_folder}/single_doc/{self.filecode}/{self.filename}_generated_queries.json'

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

            #metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            #print('\nStarting answer generation for batch of questions\n')
            sampled_entities = list(query_arr.keys())
            for ei in range(no_of_entities):
                entity = sampled_entities[ei]
                filtered_queries = query_arr[entity]
                noq = len(filtered_queries)
                print(f'total no of queries for entity {entity}: ', noq)
                #filtered_queries = [qobj for qobj in filtered_queries if "answer" not in qobj]
                for qbi in range(0, noq, self.prompt_batch_size):
                    batch_queries = filtered_queries[qbi:qbi+self.prompt_batch_size]
                    query_strs = [qobj['query'] for qobj in batch_queries]
                    query_classes = self.__classify_query(query_strs)
                    for qci, qclass in enumerate(query_classes):
                        batch_queries[qci] = batch_queries[qci] | { 'actual_query_type': qclass }
                    filtered_queries[qbi:qbi+self.prompt_batch_size] = batch_queries
                filtered_queries = [qobj for qobj in filtered_queries if len(qobj['actual_query_type']) > 0]

                query_arr[entity] = filtered_queries
                query_store["queries"] = query_arr

                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp)

                if entity in main_query_store["queries"]:
                    main_query_store["queries"][entity].extend(filtered_queries)
                else:
                    main_query_store["queries"][entity] = filtered_queries

            with open(main_query_store_fp, 'w') as fp:
                json.dump(main_query_store, fp)

            os.remove(iquery_store_fp) # remove from intermediate storage after final dataset is constructed
        else:
            SystemExit('Chunk store not found!')

if __name__ == "__main__":
    st = time()
    log_fp = f'logs/bu-answer-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filecode', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)
    parser.add_argument('--no_of_entities', type = int, default = 20, required = False)
    parser.add_argument('--query_hop_span', type = str, default = 'multi_doc', required = False)


    args = parser.parse_args()

    query_classifier = QueryClassifier(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size,
        query_hop_span = 'single_doc')
    print(f'\n\nGenerating answers for filecode: {args.filecode}')
    query_classifier.set_filename(args.filecode)
    query_classifier.classify_query(no_of_entities = args.no_of_entities)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
