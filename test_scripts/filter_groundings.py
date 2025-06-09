import json

grounding_fp = './data/chunked_data/global_chunk_store/llama/10-K_NVDA_20240128_chunk_store.json'
with open(grounding_fp, 'r') as fp:
    gobj = json.load(fp)

filtered_chunks = {"chunks": []}
for chunk in gobj["chunks"]:
    if "groundings" in chunk and len(chunk["groundings"]) > 0:
        filtered_chunks["chunks"].append(chunk)

with open('./data/chunked_data/global_chunk_store/llama/10-K_NVDA_20240128_filtered_chunk_store.json', 'w') as fp:
    json.dump(filtered_chunks, fp)
