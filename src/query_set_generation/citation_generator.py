import os
import torch
import json
from time import time, sleep
import sys
import argparse
from typing import Dict, List, Any

# custom imports
from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from src.prompts.query_set_generation.citation_prompt import CITATION_INSTRUCTION_PROMPT, CITATION_JSON_SCHEMA, CITATION_SYSTEM_PROMPT
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS
from src.base.generator import Generator

class CitationGenerator(Generator):

    def __init__(self, model_index = 6, prompt_batch_size = 3, query_hop_span: str = 'multi_hop'):
        super().__init__(model_index = model_index, prompt_batch_size = prompt_batch_size)
        self.query_hop_span = query_hop_span
        

    def __generate_citations(self, chunks: List[Dict[str, Any]], qna_pair: Dict[str, str], metadata: str):

        citation_instruction_prompts = []
        chunk_citations = [[]] * len(chunks)
        citation_system_prompt = CITATION_SYSTEM_PROMPT
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            citation_json_schema = CITATION_JSON_SCHEMA
        else:
            citation_json_schema = None

        for cobj in chunks:
            citation_instruction_prompt = CITATION_INSTRUCTION_PROMPT + f"\nText:{cobj['text']}\nQuery: {qna_pair['query']}\nAnswer: {qna_pair['answer']}\nMetadata: {metadata}"
            citation_instruction_prompts.append(citation_instruction_prompt)

        #citation_instruction_prompt = citation_instruction_prompt + f"\nText:{chunk}\nQuery: {qna_pair['query']}\nAnswer: {qna_pair['answer']}\nMetadata: {metadata}"
        csummaries, cstats = self.get_output_from_llm(citation_instruction_prompts, citation_system_prompt, citation_json_schema)

        for csi, csummary in enumerate(csummaries):
            cjson = extract_json_array_by_key(csummary, "citations")
            if cjson != None and len(cjson) > 0:
                avg_cit_tokens = cstats[csi]['output_tokens'] / len(cjson)
                citations = list(map(lambda x: { 'doc_code': chunks[csi]['doc_code'], 'chunk_index': chunks[csi]['chunk_index'], 'average_output_tokens':avg_cit_tokens, 'text': x}, cjson))
                chunk_citations[csi] = citations
            else:
                chunk_citations[csi] = []
               
        return chunk_citations

    def generate_citations(self, no_of_entities = 20):

        if self.query_hop_span == 'multi_doc':
            iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/multi_doc/multi_doc_generated_queries.json'
        else: 
            iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/single_doc/{self.filecode}/{self.filename}_generated_queries.json'
        subpar_query_store_fp = f'data/queries/{self.model_folder}/{self.filename}_subpar_queries.json'
        chunks_root_fp = 'data/chunked_data/chunks'

        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of entities formed: ', len(query_arr))

            print('\nStarting citations generation for batch of questions\n')
            sampled_entities = list(query_arr.keys())
            #sampled_entities = ["Ai", "Intangible Assets", "Data Center"]
            print('\nSampled entities: ', sampled_entities)
            less_hop_qstns = []
            citations_status_info = self.get_script_status()
            total_q_bfr = 0
            total_q_after = 0
            noe = min(no_of_entities, len(sampled_entities))
            for ei in range(noe):
                entity = sampled_entities[ei]

                if entity in citations_status_info and citations_status_info[entity] == "completed":
                    continue
                self.update_script_status(citations_status_info, entity, "ongoing")
                filtered_queries = query_arr[entity]
                print(f'total no of queries for entity {entity}: ', len(filtered_queries))
                noq = len(filtered_queries)
                total_q_bfr += noq

                for qi, query_obj in enumerate(filtered_queries):
                    chunks_used = query_obj["chunks_used"]
                    grounded_chunks = set([])
                    for cobj in chunks_used:
                        grounded_chunks.add((cobj['doc_code'], cobj['chunk_index']))

                    all_citations = []
                    cited_chunks = set([])
                    cited_docs = set([])
                    qna_pair = { 'query': query_obj['query'], 'answer': query_obj['answer'] }
                    noc = len(chunks_used)
                    docs_considered = query_obj['docs_considered']
                    metadata = f'Input source: SEC 10-K filing | Companies addressed in the input: {",".join(list(map(lambda x: COMPANY_DICT[x]["company_name"], docs_considered)))}'
                    total_citations_token_count = 0
                    for ci in range(0, noc, self.prompt_batch_size):
                        #chunk_indices = list(range(ci, min(noc, ci + self.prompt_batch_size)))
                        bchunks_used = chunks_used[ci:ci+self.prompt_batch_size]
                        bchunks = []
                        for cobj in bchunks_used:
                            cmp_filename = COMPANY_DICT[cobj['doc_code']]['filename']
                            with open(os.path.join(chunks_root_fp, f'{cmp_filename}_chunked.json'), 'r') as fp:
                                cmp_chunks = json.load(fp)["chunks"]
                            bchunks.append({"doc_code": cobj["doc_code"], "chunk_index": cobj["chunk_index"], "text": cmp_chunks[cobj["chunk_index"]]})

                        chunk_citations = self.__generate_citations(chunks = bchunks, qna_pair = qna_pair, metadata = metadata)
                        for csi, cobj in enumerate(chunk_citations):
                            if len(cobj) > 0:
                                all_citations.extend(cobj)
                                total_citations_token_count += sum(map(lambda x: x['average_output_tokens'], cobj))
                    for cobj in all_citations:
                        cited_chunks.add((cobj["doc_code"], cobj["chunk_index"]))
                        cited_docs.add(cobj["doc_code"])
                    if cited_chunks.issubset(grounded_chunks):
                        cited_docs = list(cited_docs)
                        cited_chunks = list(cited_chunks)
                    else:
                        cited_docs = []
                        cited_chunks = []
                    cited_chunks = [{"doc_code": cobj[0], "chunk_index": cobj[1]} for cobj in cited_chunks]
                    if len(cited_chunks) < 5:
                        less_hop_qstns.append({ 'entity': entity, 'query': query_obj['query'], 'answer': query_obj['answer'], 'docs_considered': cited_docs, 'groundings': query_obj['groundings'], 'citations': all_citations, 'chunks_used': cited_chunks })
                    qhop_span = 'multi_doc' if len(cited_docs) > 1 else 'single_doc'
                    filtered_queries[qi] = query_obj | { 'citations': all_citations, 'chunks_used': cited_chunks, 'docs_considered': cited_docs, 'citations_token_count': total_citations_token_count, 'query_hop_span': qhop_span }
                
                # keep only queries with citations, and those that match the intended hop span
                filtered_queries = [query_obj for query_obj in filtered_queries if "query_hop_span" in query_obj and query_obj["query_hop_span"] == self.query_hop_span and "citations" in query_obj and len(query_obj["citations"]) > 0]
                total_q_after += len(filtered_queries)
                #sleep(90)
                query_arr[entity] = filtered_queries
                query_store["queries"] = query_arr
                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp) 
                self.update_script_status(citations_status_info, entity, "completed")

            with open(subpar_query_store_fp, 'w') as fp:
                json.dump({ "queries": less_hop_qstns}, fp)
            self.reset_script_status()
            print('\nNo of questions with less than 5 hops: ', len(less_hop_qstns))
            print('\nNo of questions before citation generation: ', total_q_bfr)
            print('No of questions after citation generation: ', total_q_after)
        else:
            SystemExit('Chunk store not found!')

if __name__ == "__main__":
    st = time()
    log_fp = f'logs/bu-citation-logs.txt'
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

    cit_gen = CitationGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size, query_hop_span = 'single_doc')
    print(f'\n\nGenerating citations for filecode: {args.filecode}')
    cit_gen.set_filename(args.filecode)
    cit_gen.generate_citations(no_of_entities = args.no_of_entities)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

