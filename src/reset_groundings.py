import json
import os

chunk_store_data_path = 'data/chunked_data/global_chunk_store/llama'
chunk_store_fp = os.path.join(chunk_store_data_path, f'10-K_NVDA_20240128_chunk_store.json')
#entities_to_retain = ['Suppliers', 'Technologies']
entities_to_retain = []

with open(chunk_store_fp, 'r') as fp:
    chunk_store = json.load(fp)

for ci, chunk in enumerate(chunk_store["chunks"]):
    if "groundings" in chunk:
        groundings = chunk_store["chunks"][ci]["groundings"]
        if len(entities_to_retain) > 0:
            filtered_groundings = [grobj for grobj in groundings if grobj['entity'] in entities_to_retain]
        else:
            filtered_groundings = []
        chunk_store["chunks"][ci]["groundings"] = filtered_groundings
        #del chunk_store["chunks"][ci]["groundings"]

with open(chunk_store_fp, 'w') as fp:
    json.dump(chunk_store, fp)
