import os
import json

companies_dir = 'data/chunked_data/chunks'
company_docs = os.listdir(companies_dir)

company_maps = {}

for cd in company_docs:
    company_name = cd.split("_")[1]
    ci = cd.index('_chunked')
    company_fn = cd[:ci]
    company_maps[company_name] = {
        'filename': company_fn,
        'company_name': ''
    }

with open('./companies_dict.json', 'w') as fp:
    json.dump(company_maps, fp)
