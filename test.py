import json
import pandas as pd

with open('data/chunked_data/global_chunk_store/llama/all_doc_entity_stats.json', 'r') as fp:
    c_dict = json.load(fp)

buckets = list(range(2, 11))

for bucket in buckets:
    print(f'No of entities in bucket {bucket}: ', len(c_dict[f'{bucket}'].keys()))

bucket_size = 3
bs_entities = c_dict[f'{bucket_size}'].keys()
entity_cps = set([])

for ek in bs_entities:
    doc_groups = c_dict[f'{bucket_size}'][ek]
    for dobj in doc_groups:
        for dobj2 in dobj:
            entity_cps.add(dobj2['doc'])
print('Length of unique companies: ', len(entity_cps))
print(sorted(entity_cps))
completed_cp = set([])
for ek in bs_entities:
    entity_doc_group = c_dict[f'{bucket_size}'][ek]
    for ci, cg in enumerate(entity_doc_group):
        for cii, _ in enumerate(cg):
            if entity_doc_group[ci][cii]['groundings_generated'] == True:
                completed_cp.add(entity_doc_group[ci][cii]['doc'])

print('No of companies for which groundings are generated: ', len(completed_cp))
print(sorted(list(completed_cp)))
