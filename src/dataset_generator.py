import argparse
from time import time
import sys
import json
#from src.query_set_generation.query_generator import QueryGenerator
#from src.query_set_generation.answer_generator import AnswerGenerator
#from groundings_generator import GroundingsGenerator

# custom imports
from src.consts.company_consts import COMPANY_DICT, COMPANIES_POOL
from src.entity_generator import EntityGenerator
from src.entity_grouper import EntityGrouper

def get_script_status():
    with open('./script_status.json', 'r') as fp:
        script_status = json.load(fp)
    return script_status

def update_script_status(script_status, k, v):
    script_status[k] = v
    with open('./script_status.json', 'w') as fp:
        json.dump(script_status, fp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filecode', type = str, default = 'ALL_IN_POOL', required = False)
    parser.add_argument('--model_index', type = int, default = 11, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 2, required = False)
    parser.add_argument('--no_of_entities', type = int, default = 3, required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)
    parser.add_argument('--query_type', type = str, default = 'multi_doc', required = False)
    parser.add_argument('--query_subtype', type = str, default = 'numerical_analysis', required = False)
    parser.add_argument('--step', type = str, default = 'query_gen', required = False)
    parser.add_argument('--group_size', type = int, default = 3, required = False)

    args = parser.parse_args()

    st = time()
    log_fp = f'logs/bu-dataset-generation-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    print('Dataset Generation step: ', args.step)

    companies_pool = COMPANIES_POOL
    if args.filecode != 'ALL_IN_POOL':
        companies_pool = [cp for cp in companies_pool if cp == args.filecode]

    script_status = get_script_status()

    # entity generation step
    if args.step == 'entity_gen':
        for cp in companies_pool:
            if cp in script_status and script_status[cp] == 'completed':
                continue
            update_script_status(script_status, cp, "ongoing")
            print(f'\n\nGeneration of entities for filecode: {cp}')
            filename = COMPANY_DICT[cp]['filename']
            entity_gen = EntityGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
            entity_gen.set_filename(filename = filename, filecode = cp)
            entity_gen.generate_entities()
            update_script_status(script_status, cp, "completed")
    if args.step == 'entity_group':
        entity_grp = EntityGrouper(model_index = args.model_index)
        entity_grp.group_entities()
    else:
        print('Invalid step name passed!')
        SystemExit()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

