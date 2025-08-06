import os
import torch
import json
from time import time, sleep
import sys
import argparse
from typing import Dict, List, Any

# custom imports
from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from src.prompts.query_set_generation.answer_prompt import ANSWER_INSTRUCTION_PROMPT, ANSWER_JSON_SCHEMA
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS
from src.base.generator import Generator

class AnswerGenerator(Generator):

    def __init__(self, model_index:int = 6, prompt_batch_size:int = 3, query_hop_span: str = 'multi_doc'):
        super().__init__(model_index = model_index, prompt_batch_size = prompt_batch_size)
        self.query_hop_span = query_hop_span

    def __refine_answers(self, factoids_doc_text, factoids_arr_batch, qna_pairs, metadata):
        refineans_instruction_prompt = ANSWER_REFINEMENT_INSTRUCTION_PROMPT

        #zipped_qna_groundings = list(zip(qna_pairs, factoids_doc_texts))
        rans_instruction_prompts = [refineans_instruction_prompt + f"\nQuery: {qo['query']}\nAnswer: {qo['answer']}\nMetadata: {metadata}\nFactoids: {factoids_doc_text}" for qo in qna_pairs]
        rans_system_prompt = "You are a helpful assistant, that given a Q&A pair, a list of groundings (citations related to the given Q&A pair) improves the answer to the question based on the factoids."
        rqna_pairs = []
        #zipped_qsnts_factoids = list(zip(qna_pairs, factoids_arr_batch))
        for ri, rans_instruction_prompt in enumerate(rans_instruction_prompts):
            rasummary = self.__get_output_from_llm(rans_instruction_prompt, rans_system_prompt)
            print(f'generated response for refined answer: ', rasummary)
            ajson = extract_json_text_by_key(rasummary, "answer")
            if ajson != None and "answer" in ajson:
                if cleaned_answers[0] != "":
                    rqna_pairs.append({
                        "query": qna_pairs[ri]['query'],
                        "answer": ajson['answer'],
                        "groundings": qna_pairs[ri]['groundings']
                    })
                else:
                    rqna_pairs.append({
                        'query': qna_pairs[ri]['query'],
                        'answer': qna_pairs[ri]['answer'],
                        'groundings': qna_pairs[ri]['groundings']
                    })
            else:
                rqna_pairs.append({
                    'query': qna_pairs[ri]['query'],
                    'answer': qna_pairs[ri]['answer'],
                    'groundings': qna_pairs[ri]['groundings']
                })
        return rqna_pairs

    
    def __generate_answer(self, query_strs: List[str], rel_groundings: List[Dict[str, Any]], metadatas:List[str], entity:str):

        ans_instruction_prompts = []
        batch_count = len(query_strs)
        qng_pairs = [None] * batch_count
        for ai in range(batch_count):
            grounding_objs = [{ 'text': gr['text'], 'company_addressed': COMPANY_DICT[gr['doc_code']]['company_name'] } for gr in rel_groundings[ai]]
            groundings_str = json.dumps(
                grounding_objs,
                indent=2
            )
            ans_instruction_prompt = ANSWER_INSTRUCTION_PROMPT + f"\nQuery: {query_strs[ai]}\nEntity: {entity}\nMetadata: {metadatas[ai]}\nGroundings : {groundings_str}"
            ans_instruction_prompts.append(ans_instruction_prompt)

        ans_system_prompt = "You are a helpful assistant, that given a query and list of groundings (citations related to the query), generates meaningful answer to the question."
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            answer_json_schema = ANSWER_JSON_SCHEMA
        else:
            answer_json_schema = None
        
        asummaries, _ = self.get_output_from_llm(ans_instruction_prompts, ans_system_prompt, answer_json_schema)
        for ai in range(batch_count):
            ajson = extract_json_text_by_key(asummaries[ai], "answer")
            if ajson != None and "answer" in ajson:
                qg_pair = {
                    "query": query_strs[ai],
                    "answer": ajson["answer"],
                    "groundings": rel_groundings[ai]
                }
                qng_pairs[ai] = qg_pair
        
        return qng_pairs

    def generate_answer(self, refine_answers = False, no_of_entities = 20):

        if self.query_hop_span == "multi_doc":
            iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/multi_doc/multi_doc_generated_queries.json'
        else:
            iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/single_doc/{self.filecode}/{self.filename}_generated_queries.json'

        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)

            query_arr = query_store["queries"]

            #random_indices = random.sample(range(0, len(all_factoids)), MAX_FACTOIDS_TO_SAMPLE)
            print('\nStarting answer generation for batch of questions\n')
            sampled_entities = list(query_arr.keys())
            #sampled_entities = ["Ai", "Intangible Assets", "Data Center"]
            for ei in range(no_of_entities):
                entity = sampled_entities[ei]
                filtered_queries = query_arr[entity]
                print(f'total no of queries for entity {entity}: ', len(filtered_queries))

                #filtered_queries = [qobj for qobj in filtered_queries if "answer" not in qobj]
                noq = len(filtered_queries)
                for qi in range(0, noq, self.prompt_batch_size):
                    batch_queries = filtered_queries[qi:qi+self.prompt_batch_size]
                    query_strs = []
                    rel_groundings = []
                    rel_metadatas = []
                    for qbi, query_obj in enumerate(batch_queries):
                        #grounding_objs = [{ 'text': gr['text'], 'company_name': COMPANY_DICT[gr['doc_code']]['company_name'] } for gr in query_obj["groundings"]]
                        grounding_obj_arr = query_obj["groundings"]
                        metadata = f'Input source: SEC 10-K Filings | Companies addressed in the groundings: {",".join(list(map(lambda x: COMPANY_DICT[x]["company_name"], query_obj["docs_considered"])))}'
                        rel_groundings.append(grounding_obj_arr)
                        query_strs.append(query_obj['query'])
                        rel_metadatas.append(metadata)
                    query_objs = self.__generate_answer(query_strs = query_strs, rel_groundings = rel_groundings, metadatas=rel_metadatas, entity=entity)
                    for qbi, qobj in enumerate(query_objs):
                        if qobj != None and 'answer' in qobj:
                            batch_queries[qbi]['answer'] = qobj['answer']
                    filtered_queries[qi:qi+self.prompt_batch_size] = batch_queries
                query_arr[entity] = [query_obj for query_obj in filtered_queries if 'answer' in query_obj and query_obj['answer']]

                # NOT IMPLEMENTED
                if refine_answers:
                    pass

                query_store["queries"] = query_arr

                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp) 
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
    parser.add_argument('--refine_answers', type = bool, default = False, required = False)
    parser.add_argument('--filecode', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)
    parser.add_argument('--no_of_entities', type =int, default = 20, required = False)
    parser.add_argument('--query_hop_span', type = str, default = 'multi_doc', required = False)

    args = parser.parse_args()

    ans_gen = AnswerGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size, query_hop_span = 'single_doc')
    print(f'\n\nGenerating answers for file: {args.filecode}')
    ans_gen.set_filename(args.filecode)
    ans_gen.generate_answer(refine_answers = args.refine_answers, no_of_entities = args.no_of_entities)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
