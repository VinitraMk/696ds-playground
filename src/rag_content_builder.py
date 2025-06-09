import os
import json

FILENAMES = [
    '10-K_AMD_20231230',
    '10-K_NVDA_20240128',
    '10-K_F_20231231',
    '10-K_GM_20231231',
    '10-K_INTC_20231230',
    '10-K_TSLA_20231231'
]

if __name__ == "__main__":

    data_path = "./data/parsed_data/10-K"
    all_files = os.listdir(data_path)
    data_files = []
    for fp in FILENAMES:
        company_abbr = fp.split('_')[1]
        if company_abbr in all_files:
            full_data_path = os.path.join(data_path, f"{company_abbr}/{fp}.json")
            data_files.append(full_data_path)
    level1_keys = ['introduction', 'parti', 'partii', 'partiii', 'partiv']
    file_jsons = []
    for fpi, fp in enumerate(data_files):
        file_dict = { 'id': FILENAMES[fpi], 'text': '' }
        with open(fp, 'r') as fpp:
            json_data = json.load(fpp)
        for l1k in level1_keys:
            #print(type(json_data[l1k]))
            if l1k in json_data:
                if isinstance(json_data[l1k], dict):
                    level2_keys = json_data[l1k].keys()
                    for l2k in level2_keys:
                        file_dict['text'] = file_dict['text'] + json_data[l1k][l2k]
                else:
                    file_dict['text'] = file_dict['text'] + json_data[l1k]
        file_jsons.append(file_dict)

    with open('corpus_6.jsonl', 'w') as fp:
        for entry in file_jsons:
            json.dump(entry, fp, sort_keys=True)
            fp.write('\n')
