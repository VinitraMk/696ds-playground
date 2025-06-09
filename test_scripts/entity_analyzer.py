import argparse
import os
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default = '10-K_NVDA_20240128', type = str, required = False)

    args = parser.parse_args()

    chunk_store_fp = './data/chunked_data/global_chunk_store/llama'
    chunk_store_fn = os.path.join(chunk_store_fp, f'{args.filename}_chunk_store.json')

    if not(os.path.exists(chunk_store_fn)):
        SystemExit()

    with open(chunk_store_fn, 'r') as fp:
        chunks_obj = json.load(fp)

    chunk_arr = chunks_obj['chunks']

    entities_info = {}
    k = 10
    for chunk in chunk_arr:
        ci = chunk['chunk_index']
        chunk_ent = chunk['entities']
        for en in chunk_ent:
            if en in entities_info:
                entities_info[en]['count'] += 1
                entities_info[en]['chunk_index'].append(ci)
            else:
                entities_info[en] = {
                    'chunk_index': [ci],
                    'count': 1
                }

    ent_fn = os.path.join(chunk_store_fp, f'{args.filename}_entities_info.json')
    with open(ent_fn, 'w') as fp:
        json.dump(entities_info, fp)

    ent_vals = list(entities_info.keys())
    top_k_entities = {}
    count_vals = []

    for en in ent_vals:
        count_vals.append(entities_info[en]['count'])

    unique_counts = list(set(count_vals))
    top_k_count = unique_counts[-k:]

    for en in ent_vals:
        if entities_info[en]['count'] in top_k_count:
            top_k_entities[en] = entities_info[en]
    
    print(top_k_entities)

