import json
import os
from src.consts.company_consts import COMPANY_DICT

code = 'INTC'
chunk_store_data_path = f'data/chunked_data/global_chunk_store/llama/{code}'
fname = COMPANY_DICT[code]['filename']
chunk_store_fp = os.path.join(chunk_store_data_path, f'{fname}_chunk_store.json')
#entities_to_retain = ['Suppliers', 'Technologies']
entities_to_retain = []

with open(chunk_store_fp, 'r') as fp:
    chunk_store = json.load(fp)

for ci, chunk in enumerate(chunk_store["chunks"]):
    if "groundings" in chunk:
        groundings = chunk_store["chunks"][ci]["groundings"]
        if len(entities_to_retain) > 0:
            filtered_groundings = [grobj for grobj in groundings if grobj['entity'] in entities_to_retain]
            chunk_store["chunks"][ci]["groundings"] = filtered_groundings
            chunk_store["chunks"][ci]["groundings_versions"] = []
            chunk_store["chunks"][ci]["groundings_token_count"] = 0.0
        else:
            ckeys = list(chunk_store["chunks"][ci].keys())
            for ck in ckeys:
                if ck != "chunk_index" and ck != "entities":
                    del chunk_store["chunks"][ci][ck]

        #del chunk_store["chunks"][ci]["groundings"]

with open(chunk_store_fp, 'w') as fp:
    json.dump(chunk_store, fp)
