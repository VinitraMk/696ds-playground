import json
import pandas as pd

with open('./companies_dict.json', 'r') as fp:
    c_dict = json.load(fp)

df = pd.read_csv("nasdaq_screener.csv")

name_map = dict(zip(df["Symbol"], df["Name"]))

for symbol in name_map.keys():
    if symbol in c_dict:
        cpi = name_map[symbol]
        c_dict[symbol]['company_name'] = name_map[symbol].replace(' Common Stock', '')

print(list(map(lambda x: x[1]['company_name'], c_dict.items())))

with open('./companies_dict.json', 'w') as fp:
    json.dump(c_dict, fp)

