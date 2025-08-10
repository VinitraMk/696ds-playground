import argparse
from time import time
import sys
import json
import os
import yaml
import random
import itertools

# custom imports
from src.consts.company_consts import COMPANY_DICT, COMPANIES_POOL
from src.consts.consts import MODELS
from src.entity_generator import EntityGenerator
from src.entity_grouper import EntityGrouper
from src.groundings_generator import GroundingsGenerator
from src.query_set_generation.query_generator import QueryGenerator
from src.query_set_generation.answer_generator import AnswerGenerator
from src.query_set_generation.citation_generator import CitationGenerator
from src.query_set_generation.query_classifier import QueryClassifier

CHUNK_STORE_ROOT = 'data/chunked_data/global_chunk_store/llama'

def get_script_status():
    with open('./wrapper_script_status.json', 'r') as fp:
        script_status = json.load(fp)
    return script_status

def update_script_status(script_status, k, v):
    script_status[k] = v
    with open('./wrapper_script_status.json', 'w') as fp:
        json.dump(script_status, fp)

def reset_script():
    with open('./wrapper_script_status.json', 'w') as fp:
        json.dump({}, fp)

def get_combinations(elements, sizes):
    """
    Generate combinations of given sizes from elements.
    :param elements: list of items
    :param sizes: list of integers for combination lengths
    :return: generator of tuples
    """
    for size in sizes:
        yield from itertools.combinations(elements, size)

def check_for_groundings(bucket_size: int, filecode: str, entity: str):
    with open(doc_entity_stats_fp, 'r') as fp:
        doc_entity_mapping = json.load(fp)
    entity_doc_group = doc_entity_mapping[f'{bucket_size}'][entity]
    for ci, cg in enumerate(entity_doc_group):
        for cii, _ in enumerate(cg):
            if cg[cii]['doc'] == filecode:
                return cg[cii]['groundings_generated']
    return False

def update_groundings_stat(bucket_size: int, filecode: str):
    with open(doc_entity_stats_fp, 'r') as fp:
        doc_entity_mapping = json.load(fp)

    entity_doc_group = doc_entity_mapping[f'{bucket_size}']
    entities = list(entity_doc_group.keys())
    for ek in entities:
        entity_doc_group = doc_entity_mapping[f'{bucket_size}'][ek]
        for ci, cg in enumerate(entity_doc_group):
            for cii, _ in enumerate(cg):
                if cg[cii]['doc'] == filecode:
                    entity_doc_group[ci][cii]['groundings_generated'] = True

        doc_entity_mapping[f'{bucket_size}'][ek] = entity_doc_group

    with open(doc_entity_stats_fp, 'w') as fp:
        json.dump(doc_entity_mapping, fp)

def sample_companies(num: int = 44):
    company_names = list(COMPANY_DICT.keys())
    sampled_companies = ['NVDA', 'AMD', 'INTC', 'F', 'GM', 'TSLA'] + random.sample(company_names, num)
    with open('data/chunked_data/global_chunk_store/llama/sampled_companies.json', 'w') as fp:
        json.dump({ 'sampled_companies': sampled_companies }, fp)


if __name__ == "__main__":

    try:
        with open('dataset_gen_param.yaml', 'r') as file:
            args = yaml.safe_load(file)
            print(args)
            #print(f"Database host: {data['database']['host']}")
            #print(f"Debug mode: {data['settings']['debug_mode']}")
    except FileNotFoundError:
        print("Error: 'dataset_gen_param.yaml' not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--filecode', type = str, default = 'ALL_IN_POOL', required = False)
    parser.add_argument('--model_index', type = int, default = 11, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 2, required = False)
    parser.add_argument('--no_of_entities', type = int, default = 3, required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)
    parser.add_argument('--query_hop_span', type = str, default = 'multi_doc', required = False)
    parser.add_argument('--query_type', type = str, default = 'all', required = False)
    parser.add_argument('--step', type = str, default = 'query_gen', required = False)
    parser.add_argument('--bucket_size', type = int, default = 2, required = False)
    parser.add_argument('--skip_entity_sampling', type = bool, default = False, required = False)
    parser.add_argument('--use_bucket_entities', type = bool, default = True, required = False)

    args = parser.parse_args()
    '''

    st = time()
    log_fp = f'logs/bu-dataset-generation-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    print('Dataset Generation step: ', args['step'])

    companies_pool = COMPANIES_POOL
    if args["filecode"] != 'ALL_IN_POOL':
        companies_pool = [cp for cp in companies_pool if cp == args["filecode"]]

    script_status = get_script_status()
    model_name = MODELS[args['model_index']]
    if "llama" in model_name.lower():
        model_folder = "llama"

    entity_stats_path = f'data/chunked_data/global_chunk_store/{model_folder}'
    doc_entity_stats_fp = os.path.join(entity_stats_path, 'all_doc_entity_stats.json')
    doc_groups_fp = os.path.join(entity_stats_path, 'sampled_doc_groups.json')
    sampled_entities_md_fp = f'data/chunked_data/global_chunk_store/{model_folder}/multi_doc_sampled_entities.json'

    # entity generation step
    if args['step'] == 'setup':
        sample_companies()
    elif args['step'] == 'entity_gen':
        for cp in companies_pool:
            if cp in script_status and script_status[cp] == 'completed':
                continue
            update_script_status(script_status, cp, "ongoing")
            print(f'\n\nGeneration of entities for filecode: {cp}')
            entity_gen = EntityGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'])
            entity_gen.set_filename(filecode = cp)
            entity_gen.generate_entities()
            update_script_status(script_status, cp, "completed")
        reset_script()
    elif args['step'] == 'entity_group':
        entity_grp = EntityGrouper(model_index = args['model_index'])
        entity_grp.group_entities()
    elif args['step'] == 'groundings_gen':
        if args['query_hop_span'] == 'multi_doc':
            with open(doc_entity_stats_fp, 'r') as fp:
                doc_entity_mapping = json.load(fp)
            bucket_groups = doc_entity_mapping[f'{args["bucket_size"]}']
            
            
            if os.path.exists(sampled_entities_md_fp):
                with open(sampled_entities_md_fp, 'r') as fp:
                    sentities = json.load(fp)
                if f'{args["bucket_size"]}' in sentities:
                    bucket_entities = sentities[f'{args["bucket_size"]}']
                else:
                    ents = list(bucket_groups.keys())
                    bucket_entities = random.sample(ents, args['no_of_entities'])
            else:
                if args['no_of_entities'] > len(bucket_groups.keys()):
                    ents = list(bucket_groups.keys())
                    bucket_entities = random.sample(ents, args['no_of_entities'])
                else:
                    bucket_entities = list(bucket_groups.keys())
                sentities = {}
            sentities[f'{args["bucket_size"]}'] = bucket_entities
            with open(sampled_entities_md_fp, 'w') as fp:
                json.dump(sentities, fp)

            #doc_entity_grounding_stat = {}
            for ek in bucket_entities:
                script_status = get_script_status()
                for grp in bucket_groups[ek]:
                    for doc in grp:
                        if check_for_groundings(args['bucket_size'], doc['doc'], ek):
                            continue
                        update_script_status(script_status, doc['doc'], 'ongoing')
                        print(f'\n\nGenerating groundings for filecode: {doc["doc"]}')
                        ground_gen = GroundingsGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'])
                        ground_gen.set_filename(filecode = doc["doc"])
                        ground_gen.generate_groundings(skip_entity_sampling = args['skip_entity_sampling'],
                            use_bucket_entities = args['use_bucket_entities'],
                            no_of_entities = args['no_of_entities'],
                            bucket_size = args['bucket_size'])
                        update_script_status(script_status, doc['doc'], 'completed')
                        update_groundings_stat(args['bucket_size'], doc['doc'])
                                
                reset_script()
        else:
            print(f'\n\nGenerating groundings for filecode: {args["filecode"]}')
            ground_gen = GroundingsGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'])
            ground_gen.set_filename(filecode = args["filecode"])
            ground_gen.generate_groundings(skip_entity_sampling = args['skip_entity_sampling'],
                use_bucket_entities = args['use_bucket_entities'],
                no_of_entities = args['no_of_entities'],
                bucket_size = args['bucket_size'])
    elif args['step'] == "query_gen":
        if args['query_hop_span'] == "multi_doc":
            doc_groups_info = {}
            with open(doc_entity_stats_fp, 'r') as fp:
                doc_entity_mapping = json.load(fp)

            with open(sampled_entities_md_fp, 'r') as fp:
                sampled_entities_info = json.load(fp)
            
            bucket_groups = doc_entity_mapping[f'{args["bucket_size"]}']
            bucket_entities = list(bucket_groups.keys())
            if args['no_of_entities'] != -1:
                #bucket_entities = bucket_entities[:args['no_of_entities']]
                bucket_entities = sampled_entities_info[f'{args["bucket_size"]}']
            for bek in bucket_entities:
                doc_groups = bucket_groups[bek]
                dcg_codes = [list(map(lambda x: x['doc'], dcg)) for dcg in doc_groups]
                doc_groups_info[bek] = dcg_codes
            with open(doc_groups_fp, 'w') as fp:
                json.dump(doc_groups_info, fp)
                    
            query_gen = QueryGenerator(model_index = args['model_index'],
                query_type = args['query_type'], query_hop_span = args['query_hop_span'])
            print(f'\n\nGenerating queries for file: {args["filecode"]}')
            query_gen.set_filename('NVDA')
            query_gen.generate_query(
                no_of_qstns = args['no_of_qstns'],
                no_of_entities = args['no_of_entities'] if args['no_of_entities'] != -1 else len(bucket_entities))
        else:
            query_gen = QueryGenerator(model_index = args['model_index'],
                query_type = args['query_type'], query_hop_span = args['query_hop_span'])
            print(f'\n\nGenerating queries for file: {args["filecode"]}')
            query_gen.set_filename(args["filecode"])
            query_gen.generate_query(no_of_qstns = args['no_of_qstns'],
                no_of_entities = args['no_of_entities'])
    elif args['step'] == "answer_gen":
        if args['query_hop_span'] == "multi_doc":
            with open(doc_groups_fp, 'r') as fp:
                doc_groups_info = json.load(fp)
            noe = len(doc_groups_info)
            ans_gen = AnswerGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'],
                query_hop_span = args['query_hop_span'])
            print(f'\n\nGenerating answers for file: {args["filecode"]}')
            ans_gen.set_filename('NVDA')
            ans_gen.generate_answer(refine_answers = False, no_of_entities = noe)
        else:
            ans_gen = AnswerGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'],
                query_hop_span = args['query_hop_span'])
            print(f'\n\nGenerating answers for file: {args["filecode"]}')
            ans_gen.set_filename('NVDA')
            ans_gen.generate_answer(refine_answers = False, no_of_entities = args['no_of_entities'])
    elif args['step'] == "citation_gen":
        if args['query_hop_span'] == "multi_doc":
            with open(doc_groups_fp, 'r') as fp:
                doc_groups_info = json.load(fp)
            noe = len(doc_groups_info)
            cit_gen = CitationGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'], query_hop_span = 'multi_doc')
            print(f'\n\nGenerating answers for filecode: {args["filecode"]}')
            cit_gen.set_filename('NVDA')
            cit_gen.generate_citations(no_of_entities = noe)
        else:
            cit_gen = CitationGenerator(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'], query_hop_span = 'single_doc')
            print(f'\n\nGenerating answers for filecode: {args["filecode"]}')
            cit_gen.set_filename(args["filecode"])
            cit_gen.generate_citations(no_of_entities = args['no_of_entities'])
    elif args['step'] == "query_classification":
        if args['query_hop_span'] == "multi_doc":
            with open(doc_groups_fp, 'r') as fp:
                doc_groups_info = json.load(fp)
            noe = len(doc_groups_info)
            query_classifier = QueryClassifier(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'],
                query_hop_span = 'multi_doc')
            print(f'\n\nGenerating answers for filecode: {args["filecode"]}')
            query_classifier.set_filename('NVDA')
            query_classifier.classify_query(no_of_entities = noe)
        else:
            query_classifier = QueryClassifier(model_index = args['model_index'], prompt_batch_size = args['prompt_batch_size'],
                query_hop_span = 'single_doc')
            print(f'\n\nGenerating answers for filecode: {args["filecode"]}')
            query_classifier.set_filename(args["filecode"])
            query_classifier.classify_query(no_of_entities = args['no_of_entities'])
    else:
        print('Invalid step name passed!')
        SystemExit()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

