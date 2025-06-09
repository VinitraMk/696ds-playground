import os
import json

if __name__ == "__main__":

    queries_path = "./data/queries/llama"
    chunk_store_path = "./data/chunked_data/global_chunk_store/qwq"
    queries_json_files = [f for f in os.listdir(queries_path) if ".json" in f]
    chunk_json_files = [f for f in os.listdir(chunk_store_path) if ".json" in f]

    SEED_METADATA_TOPICS = [
        "Risk Factors and Challenges",
        "Financial Performance and Metrics",
        "Business Operations, Strategy, and Market Positioning",
        "Market Trends, Economic Environment, and Industry Dynamics"
    ]

    FILENAMES = [
        '10-K_AMD_20231230',
        '10-K_NVDA_20240128',
        '10-K_F_20231231',
        '10-K_GM_20231231',
        '10-K_INTC_20231230',
        '10-K_TSLA_20231231'
    ]

    for fn in FILENAMES:
        print('\nProcessing file: ', fn)
        query_fpath = os.path.join(queries_path, f'{fn}_gen_queries.json')
        chunk_path = os.path.join(chunk_store_path, f'{fn}_chunk_store.json')
        with open(query_fpath, 'r') as fp:
            query_store = json.load(fp)
        with open(chunk_path, 'r') as fp:
            chunk_store = json.load(fp)

        for topic in SEED_METADATA_TOPICS:
            qkeys = [qobj['topic'] for qobj in query_store['queries']]
            if topic in qkeys:
                topic_obj = [(tq,ti) for ti, tq in enumerate(query_store['queries']) if tq['topic'] == topic]
                topic_queries = topic_obj[0][0]
                ti = topic_obj[0][1]
                qsets = topic_queries['query_sets']

                for qi,qs in enumerate(qsets):
                    qs['chunk_indices'] = []
                    grs = qs['groundings']

                    for chunk in chunk_store['chunks']:
                        topic_factoids = [fobj for fobj in chunk['factoids'] if fobj['topic'] == topic]
                        ci = chunk['chunk_index']

                        for fobj in topic_factoids:
                            if (fobj['citation'] in grs) and (ci not in qs['chunk_indices']):
                                query_store['queries'][ti]['query_sets'][qi]['chunk_indices'].append(ci)
                                #qs['chunk_indices'].append(ci)
                #updated_query_path = os.path.join(queries_path, f'{fn}_gen_queries_updated.json')
                with open(query_fpath, 'w') as fp:
                    json.dump(query_store, fp)
                
                    

