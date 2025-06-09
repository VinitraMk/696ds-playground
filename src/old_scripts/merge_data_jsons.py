import argparse
import os
import json

MAIN_DATA_PATH = 'data/queries'
SEED_METADATA_TOPICS = [
    "Risk Factors and Challenges",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
    "Market Trends, Economic Environment, and Industry Dynamics"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default = 'gemini', required = False)

    args = parser.parse_args()

    data_path = os.path.join(MAIN_DATA_PATH, args.data_folder)
    full_data = {
        'queries': []
    }
    if os.path.exists(data_path):
        all_json_files = [f for f in os.listdir(data_path) if (".json" in f and f != "full_dataset.json")]
    dataset_len = 0
    for topic in SEED_METADATA_TOPICS:
        topic_queries = []
        for jf_name in all_json_files:
            jf_path = os.path.join(data_path, jf_name)
            with open(jf_path, 'r') as fp:
                jobj = json.load(fp)
            filtered_query_sets = [qo for qo in jobj["queries"] if qo['topic'] == topic]
            if len(filtered_query_sets) > 0:
                filtered_query_sets = filtered_query_sets[0]
            topic_queries.extend(filtered_query_sets['query_sets'])
        full_data['queries'].append({
            'topic': topic,
            'query_sets': topic_queries
        })
        dataset_len += len(topic_queries)
    with open(os.path.join(data_path, 'full_dataset.json'), 'w') as fp:
        json.dump(full_data, fp)
    print('No of data points: ', dataset_len)
            
            



    