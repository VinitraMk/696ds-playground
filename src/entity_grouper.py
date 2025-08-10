from time import time
import sys
import argparse
import json
import os
import matplotlib.pyplot as plt
from typing import List
import itertools

#custom imports
from src.consts.company_consts import COMPANY_DICT, COMPANIES_POOL
from src.consts.consts import MODELS

def get_combinations(elements, sizes):
    """
    Generate combinations of given sizes from elements.
    :param elements: list of items
    :param sizes: list of integers for combination lengths
    :return: generator of tuples
    """
    for size in sizes:
        yield from itertools.combinations(elements, size)

class EntityGrouper:

    def __init__(self, model_index: int = 11):
        self.model_name = MODELS[model_index]
        if "llama" in self.model_name.lower():
            self.model_folder = "llama"
        else:
            print('Invalid model index passed!')
            SystemExit()

    def identify_common_entities(self, companies_to_anchor: List[str]):
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
        optimal_entities = []
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
                        f'{cp}': {
                            'count': company_entities_info[ek]['count'],
                            'chunk_indices': company_entities_info[ek]['chunk_indices']
                        }
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
                optimal_entities.append(ek)

        print('Entities in optimal range count (5, 20): ', optimum_range_count)
        with open('./company_common_entities_info.json', 'w') as fp:
            json.dump(company_common_entities_info, fp)
        with open('./sampled_common_entities.json', 'w') as fp:
            json.dump({'docs_considered': companies_to_anchor, 'common_entities': optimal_entities}, fp)

    def get_entity_doc_mapping(self, companies_pool: List[str]):
        all_comp_entities = {}
        all_comp_entity_keys = {}
        data_folder_path = f'data/chunked_data/global_chunk_store/{self.model_folder}'
        doc_buckets = list(range(2, min(len(companies_pool), 10)+1))
        print('Document buckets to be considered: ', doc_buckets)

        all_entities = []
        entities_to_doc = {}
        for cp in companies_pool:
            data_path = os.path.join(data_folder_path, f'{cp}/{COMPANY_DICT[cp]["filename"]}_entities_info.json')
            with open(data_path, 'r') as fp:
                entities_info = json.load(fp)
            #all_comp_entities[cp] = list(entities_info.keys())
            #all_comp_entities = entities_info
            all_comp_entity_keys[cp] = list(entities_info.keys())
            all_entities.extend(list(entities_info.keys()))

        all_entities = list(set(all_entities))
        entities_to_doc_chunk = {}
        for ek in all_entities:
            entities_to_doc[ek] = { 'count': 0, 'docs': [], 'good_chunks_count': 0 }
            for cp in companies_pool:
                if ek in all_comp_entity_keys[cp]:
                    data_path = os.path.join(data_folder_path, f'{cp}/{COMPANY_DICT[cp]["filename"]}_entities_info.json')
                    with open(data_path, 'r') as fp:
                        entities_info = json.load(fp)

                    entities_to_doc[ek]['count'] += 1
                    entities_to_doc[ek]['docs'].append({
                        'doc': cp,
                        'chunk_count': entities_info[ek]['count'],
                        'chunk_indices': entities_info[ek]['chunk_indices'],
                        'groundings_generated': False
                    })

        '''
        for bs in range(1, len(companies_pool)+1):
            for ek in all_entities:
                if all_entities[ek]['count'] == bs:
                    for cp in companies_pool:
                        if ek in all_comp_entity_keys[cp]:
                            data_path = os.path.join(data_folder_path, f'{cp}/{COMPANY_DICT[cp]["filename"]}_entities_info.json')
                            with open(data_path, 'r') as fp:
                                entities_info = json.load(fp)
                            entities_to_doc[ek]['good_chunks_count'] += (entities_info[ek]['count'] >= 5 and entities_info[ek] <= 15)
        '''            

        doc_hist = [0] * len(companies_pool)
        entity_doc_vals = list(map(lambda x: x['count'], list(entities_to_doc.values())))
        for c in range(len(companies_pool)):
            co = entity_doc_vals.count(c+1)
            doc_hist[c] = co
        print('doc hist keys', doc_hist)
        plt.bar(doc_buckets, doc_hist[1:10])
        plt.xticks(doc_buckets)
        print('Document bar plot values: ', doc_hist)
        plt.xlabel('No of doc buckets')
        plt.ylabel('No of entities in doc buckets')
        plt.title('Entities to doc count (without hop filter)')
        plt.savefig('data/plots/entities_to_doc_hist.png')
        plt.clf()

        '''
        doc_hist = [0] * len(companies_pool)
        entity_doc_vals = list(map(lambda x: x['good_chunks_count'], list(entities_to_doc.values())))
        for c in range(len(companies_pool)):
            co = entity_doc_vals.count(c+1)
            doc_hist[c] = co

        plt.bar(doc_buckets, doc_hist[1:])
        plt.xticks(doc_buckets)
        print('Document bar plot values: ', doc_hist)
        plt.xlabel('No of doc buckets')
        plt.ylabel('No of entities in doc buckets')
        plt.title('Entities to doc count (without hop filter)')
        plt.savefig('data/plots/entities_to_doc_hist.png')
        plt.clf()
        '''

        doc_to_entities = {}
        
        for ek in all_entities:
            for bucket in doc_buckets:
                if bucket not in doc_to_entities:
                    doc_to_entities[bucket] = {}
                if ek in entities_to_doc and entities_to_doc[ek]['count'] == bucket:
                    #print('entities: ', ek, entities_to_doc[ek])
                    optimal_bucket_entities = [dobj for dobj in entities_to_doc[ek]['docs']]
                    optimal_count = list(map(lambda x: x['chunk_count'], optimal_bucket_entities))
                    found = False
                    if len(optimal_bucket_entities) == bucket and sum(optimal_count) >= 5 and sum(optimal_count) <= 20:
                        if ek in doc_to_entities[bucket]:
                            doc_to_entities[bucket][ek].append(optimal_bucket_entities)
                        else:
                            doc_to_entities[bucket][ek] = [optimal_bucket_entities]
                        found = True
                    if found:
                        break
                    

        with open(f'data/chunked_data/global_chunk_store/{self.model_folder}/all_doc_entity_stats.json', 'w') as fp:
            json.dump(doc_to_entities, fp)

    def group_entities(self):
        #companies_pool = ['NVDA', 'AMD', 'INTC', 'TSLA', 'F', 'GM']
        #companies_to_anchor = ['NVDA', 'AMD', 'INTC']
        companies_pool = COMPANIES_POOL
        #self.identify_common_entities(companies_to_anchor)
        self.get_entity_doc_mapping(companies_pool)
        
        
if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-entity-grouping-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type = int, default = 11, required = False)

    args = parser.parse_args()

    entity_grouper = EntityGrouper(model_index = args.model_index)
    entity_grouper.group_entities()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
