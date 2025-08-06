import os
import sys
import json
from time import time, sleep
import argparse
import matplotlib.pyplot as plt
from typing import List, Any

#custom imports
from utils.string_utils import extract_json_array_by_key, is_valid_sentence, extract_json_object_array_by_keys, extract_json_text_by_key
from src.prompts.entity_generation.entity_prompt import ENTITY_INSTRUCTION_PROMPT, ENTITY_SYSTEM_PROMPT, ENTITY_JSON_SCHEMA
from src.consts.company_consts import COMPANY_DICT
from src.base.generator import Generator

class EntityGenerator(Generator):

    def __init__(self, model_index: int = 11, prompt_batch_size: int = 1):
        super().__init__(model_index, prompt_batch_size)
        

    def __generate_entities_from_chunks(self, chunks: List[Any]):

        entity_instruction_prompt = ENTITY_INSTRUCTION_PROMPT
        entity_json_schema = ENTITY_JSON_SCHEMA
        entity_system_prompt = ENTITY_SYSTEM_PROMPT
        chunk_entities = []
        entity_info = {}
        entity_chunk_info = {}
        chunk_count = len(chunks)

        for ci in range(0, chunk_count, self.prompt_batch_size):
            batch_chunks = chunks[ci:ci+self.prompt_batch_size]
            entity_prompt_texts = [entity_instruction_prompt + f"\nChunk: {chunk}" for chunk in batch_chunks]
            #entity_prompt_text = 
            esummaries, _ = self.get_output_from_llm(entity_prompt_texts, entity_system_prompt, entity_json_schema)
            ci_indices = list(range(ci, min(ci+self.prompt_batch_size, chunk_count)))
            for cbii, cbi in enumerate(ci_indices):
                print('esummaries: ', esummaries, cbii)
                ejson = extract_json_array_by_key(esummaries[cbii], "entities")
                if ejson != None and len(ejson) > 0:
                    ejson = list(set([et.upper() for et in ejson]))
                    print(f'Entities from chunk {cbi}:', ejson)
                    for en in ejson:
                        if en in entity_info:
                            entity_info[en] = entity_info[en] + 1
                            entity_chunk_info[en].append(cbi)
                        else:
                            entity_info[en] = 1
                            entity_chunk_info[en] = [cbi]
                    chunk_entities.append(ejson)
                else:
                    chunk_entities.append([])
        #print('chunk entities', chunk_entities)
        count_vals = list(entity_info.values())
        count_keys = list(entity_info.keys())
        entity_vks = sorted(list(zip(count_vals, count_keys)), reverse=True)
        sorted_entity_info = {}
        for el in entity_vks:
            if entity_info[el[1]] == el[0]:
                sorted_entity_info[el[1]] = { 'count': entity_info[el[1]], 'chunk_indices': entity_chunk_info[el[1]] }

        return chunk_entities, sorted_entity_info
    
    def generate_entities(self):

        raw_chunks_fp = 'data/chunked_data/chunks'
        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filecode}'
        plot_stats_fp = f'figures/data/queries/{self.model_folder}/{self.filecode}'
        os.makedirs(plot_stats_fp, exist_ok = True)
        os.makedirs(chunk_store_fp, exist_ok = True)

        chunk_fn = f'{self.filename}_chunked'
        chunk_fp = f'data/chunked_data/chunks/{chunk_fn}.json'
        with open(chunk_fp, 'r') as fp:
            self.all_chunks = json.load(fp)["chunks"]

        self.chunks = [{ 'chunk_index': ci, 'text': self.all_chunks[ci] } for ci in range(len(self.all_chunks))]
        print('No of chunks: ', len(self.chunks))

        file_chunkstore_fp = os.path.join(chunk_store_fp, f'{self.filename}_chunk_store.json')
        if os.path.exists(file_chunkstore_fp):
            with open(file_chunkstore_fp, 'r') as fp:
                chunks_obj = json.load(fp)
        else:
            chunks_obj = { "chunks": [] }

        all_resp = []
        chunk_entities, entity_info = self.__generate_entities_from_chunks(self.chunks)

        entity_count_values = [eiob['count'] for eiob in list(entity_info.values())]
        entity_values = list(entity_info.keys())
        #plt.bar(entity_values, entity_count_values)
        #plt.savefig(f'{plot_stats_fp}/{self.filename}_entity_dist.png')

        # save entities info
        with open(f'{chunk_store_fp}/{self.filename}_entities_info.json', 'w') as fp:
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

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 11, required = False)
    parser.add_argument('--filecode', type=str, default = 'NVDA', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)

    args = parser.parse_args()

    entity_gen = EntityGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    entity_gen.set_filename(args.filecode)
    entity_gen.generate_entities()
    
    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
