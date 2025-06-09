import torch
import os
import sys
import json
from time import time
from vllm import LLM
import argparse
import multiprocessing
import re
import ast
from utils.string_utils import extract_json_array_by_key, is_valid_sentence, extract_json_object_array_by_keys, extract_json_text_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, get_tokenizer

COMPANY_DICT = {
    'INTC': 'Intel Corp.',
    'AMD': 'AMD Inc.',
    'NVDA': 'Nvidia Corp.',
    'TSLA': 'Tesla Inc.',
    'F': 'Ford Motor Company',
    'GM': 'General Motors'
}

MODELS = [
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    "Qwen/QwQ-32B-AWQ",
    "meta-llama/Meta-Llama-3-70B"
]

INSTRUCTION_TYPES = {
    'FACTOIDS': 'factoids',
    'QUERIES': 'queries',
    'METADATA': 'metadata',
    'CHUNK_CLASSIFICATION': 'chunk_classification'
}

SEED_METADATA_TOPICS = [
    "Risk Factors and Challenges",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
    "Market Trends, Economic Environment, and Industry Dynamics"
]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR

RELEVANCE_THRESHOLD = 2.0
CHUNK_BATCH_SIZE = 5

class FactoidGen:

    def __init__(self, filename, model_index = 0):
        self.filename = filename
        self.model_index = model_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f'Device enabled: {self.device}')

        # Load the model using vLLM
        self.model_name = MODELS[self.model_index]
        print('Model used: ', self.model_name)
        self.tokenizer = get_tokenizer(model_name = self.model_name)

        if "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization = 0.95,
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        else:
            print('Invalid model name passed!')
            SystemExit()


    def __extract_and_clean_factoids(self, text):
        #factoid_list = extract_json_object_array_by_keys(text, ["factoid", "citation"])
        factoid_list = extract_json_array_by_key(text, "factoids")
        if (factoid_list) and (len(factoid_list) > 0):
        #print('factoid list', factoid_list)
            clean_factoids_citations = [s for s in factoid_list if is_valid_sentence(s)]
        else:
            clean_factoids_citations = []
        return clean_factoids_citations

    def __generate_entities_from_factoids(self, factoids):

        entity_instruction_prompt = """
        Given a factoid, identify all significant entites or "nouns" described in each of the factoids.
        This should include but not limited to:
        - Object: Any concrete object that is referenced by the provided content.
        - Organization: Any organization working with the main company either on permanent or temporary basis on some contracts.
        - Concepts: Any significant abstract ideas or themes that are central to the factoids.

        ### Input Format:
        - Factoid: <factoid text>

        ### Output Format (JSON):
        "entities": ['entity 1', 'entity 2', ...]

        ### Input for your task:
        """
        entity_system_prompt = "You are a helpful assistant, that given a list of factoids, generates entites addressed in the factoids."

        #factoid_prompts = { f'{ci_str}': entity_instruction_prompt + "\nFactoids: [" + ",\n".join([f"{item['factoid']}" for item in factoids[ci_str]]) + "]" for ci_str in factoids.keys()}
        #prompt_tokens = [get_prompt_token(p, entity_system_prompt, self.tokenizer) for p in factoid_prompts.values()]

        for ci, ci_str in enumerate(factoids.keys()):
            for fi, fobj in enumerate(factoids[ci_str]):
                factoid_prompt = entity_instruction_prompt + f"\nFactoid: {fobj['factoid']}"
                prompt_token = [get_prompt_token(factoid_prompt, entity_system_prompt, self.tokenizer)]
                eoutputs = execute_LLM_tasks(self.llm, prompt_token, max_new_tokens = 4096, temperature = 0.6, top_p = 0.9)
                esummary = eoutputs[0].outputs[0].text.strip()
                print('generated response: ', esummary)
                out = extract_json_array_by_key(esummary, "entities")
                if out:
                    print('extracted response', out)
                    factoids[ci_str][fi]['entities'] = out
                else:
                    factoids[ci_str][fi]['entities'] = []
        return factoids
    
    def __generate_citations_for_factoids(self, chunks, factoids):

        citation_instruction_prompt = """
        Given a factoid, and chunk of text identify and extract the exact sentence or passage from the chunk of text (citation) that was used to construct the factoid.
        
        ### Input Format:
        - Text: <chunk of text>
        - Factoid: <factoid text>

        ### Output Format (JSON):
        {
            "citation": <citation_text>
        }

        ### Input for your task:
        """
        citation_system_prompt = "You are a helpful assistant, that given a factoid and a chunk, extracts citation from the chunk of text used to make the factoid."

        for ci,co in enumerate(chunks):
            ci_str = str(co['chunk_index'])
            print('res ci', type(ci_str))
            if ci_str in factoids:
                print('res factoids', factoids[ci_str])
                for fi, fobj in enumerate(factoids[ci_str]):
                    factoid_prompt = citation_instruction_prompt + f"\nText: {co['text']}\nFactoid: {fobj['factoid']}"
                    prompt_token = [get_prompt_token(factoid_prompt, citation_system_prompt, self.tokenizer)]
                    coutputs = execute_LLM_tasks(self.llm, prompt_token, max_new_tokens = 4096, temperature = 0.6, top_p = 0.9)
                    esummary = coutputs[0].outputs[0].text.strip()
                    print('generated response: ', esummary)
                    out = extract_json_text_by_key(esummary, "citation")
                    if out:
                        print('extracted response', out)
                        factoids[ci_str][fi]['citation'] = out['citation']
                    else:
                        factoids[ci_str][fi]['citation'] = ''
        return factoids

    def __generate_factoids_from_all_chunks(self, chunks, topic):
        instruction_prompt = """
        Given a text and a topic, extract verifiable factoids from it related to the topic.  A factoid is a discrete, factual statement about the topic.

        ### Generation guidelines:
        - **No opinions, adjectives, elaborations or extra details**
        - Each factoid must be standalone, verifiable statement.
        - Extract any numeric information which is relevant. If there is data in a text based table, extract the relevant data.
        - Use concise, standalone statements, in English.
        - Focus only on information related to the provided topic.
        - Don't think for more than 3000 tokens.
        - If you can't find any factoids, return an empty list presented like this "factoids: []".

        ### Input Format:
        - Topic: <topic>
        - Text: <text>

        ### Output Format (JSON):
        "factoids": [
            "Factoid 1", 
            "Factoid 2"
        ]

        ### Example:
        Topic: "Litigation"
        Text Chunk:
        "Company X is currently involved in a class action lawsuit filed in March 2023 concerning alleged violations of securities laws. The case is pending in the U.S. District Court for the Southern District of New York. No financial settlement has been reached as of the filing date."

        Output (JSON):
        "factoids": [
            "Company X is facing a class action lawsuit related to alleged securities law violations.",
        ]
        
        ### Now process the input:
        """

        all_res = {}
        factoid_system_prompt = 'You are a helpful assistant that extracts factoids from text.'

        for i in range(0, len(chunks), CHUNK_BATCH_SIZE):
            print(f'\nProcessing chunk batch {i}')
            chunk_batch = chunks[i:(i+CHUNK_BATCH_SIZE)]
            chunk_to_prompts = {f'{chunk["chunk_index"]}': instruction_prompt + f"\nTopic: {topic}\nText chunk: {chunk['text']}" for chunk in chunk_batch}
            prompt_tokens = [get_prompt_token(p, factoid_system_prompt, self.tokenizer) for p in chunk_to_prompts.values()]
            outputs = execute_LLM_tasks(self.llm, prompt_tokens, max_new_tokens=4096, temperature=0.7, top_p=0.9)
            res = {}
            for j, o in zip(chunk_to_prompts.keys(), outputs):
                print(f'generated response for chunk {j}: ', o.outputs[0].text.strip())
                out = self.__extract_and_clean_factoids(o.outputs[0].text.strip())
                if out:
                    res[j] = [{ 'factoid': fstr } for fstr in out]
            print('resulting factoids', res, len(res.keys()))
            all_res = all_res | res

        print('dict len', len(all_res.keys()))
        all_res = self.__generate_citations_for_factoids(chunks, all_res)
        all_res = self.__generate_entities_from_factoids(all_res)
        return all_res

    def generate_factoids(self, topic_index):

        chunk_fn = f'{self.filename}_chunked'
        chunk_fp = f'data/chunked_data/{chunk_fn}.json'
        scored_chunk_fp = f'data/chunked_data/scored_chunks/{chunk_fn}.json'
        with open(chunk_fp, 'r') as fp:
            self.all_chunks = json.load(fp)["chunks"]
        with open(scored_chunk_fp, 'r') as fp:
            self.scored_chunks = json.load(fp)
        self.chunks = [{ 'chunk_index': ci, 'text': self.all_chunks[ci] } for ci in range(len(self.scored_chunks)) if self.scored_chunks[ci]["relevance"][topic_index] >= RELEVANCE_THRESHOLD]
        print('Filtered chunks: ', len(self.chunks))

        file_chunkstore_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
        if os.path.exists(file_chunkstore_fp):
            with open(file_chunkstore_fp, 'r+') as fp:
                chunks_obj = json.load(fp)
            chunk_arr = chunks_obj["chunks"] 
        else:
            chunks_obj = { "chunks": [] }
            chunk_arr = []

        all_resp = []
        chunk_factoids = self.__generate_factoids_from_all_chunks(self.chunks, SEED_METADATA_TOPICS[topic_index])
        #print(chunk_factoids)
        all_resp = []
        for i in range(len(self.all_chunks)):

            if f'{i}' in chunk_factoids:
                if len(chunk_arr) > 0:
                    existing_topics = chunk_arr[i]["topics"]
                    existing_factoids = chunk_arr[i]["factoids"]
                else:
                    existing_topics = [SEED_METADATA_TOPICS[topic_index]] if len(chunk_factoids[f'{i}']) > 0 else []
                    existing_factoids = []
                if SEED_METADATA_TOPICS[topic_index] not in existing_topics:
                    existing_topics.append(SEED_METADATA_TOPICS[topic_index])
                chunk_facts = [{ 'topic': SEED_METADATA_TOPICS[topic_index], 'factoid': factoid_cit['factoid'], 'citation': factoid_cit['citation'], 'entities': factoid_cit['entities'] } for factoid_cit in chunk_factoids[f'{i}'] if ((len(factoid_cit['entities']) > 0) and (factoid_cit['citation'] != ""))]
                existing_factoids.extend(chunk_facts)
                chunk_resp = {
                    'chunk_index': i,
                    'factoids': existing_factoids,
                    'topics': existing_topics
                }
            else:
                if len(chunk_arr) > 0:
                    existing_topics = chunk_arr[i]["topics"]
                    existing_factoids = chunk_arr[i]["factoids"]
                else:
                    existing_factoids = []
                    existing_topics = []
                chunk_resp = {
                    'chunk_index': i,
                    'factoids': existing_factoids,
                    'topics': existing_topics
                }
            all_resp.append(chunk_resp)
        chunks_obj["chunks"] = all_resp
        if os.path.exists(file_chunkstore_fp):
            with open(file_chunkstore_fp, 'w') as fp:
                json.dump(chunks_obj, fp)
        else:
            fp = open(file_chunkstore_fp, 'x')
            json.dump(chunks_obj, fp)

        


if __name__ == "__main__":
    
    st = time()
    log_fp = f'logs/bu-factoid-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--filename', type=str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    #filename = '10-K_NVDA_20240128'
    print('topic index in args', args.topic_index, type(args.topic_index))

    factoid_gen = FactoidGen(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(len(SEED_METADATA_TOPICS)):
            print(f'Generating factoids based on topic: {SEED_METADATA_TOPICS[ti]}')
            factoid_gen.generate_factoids(ti)
            print(f'\n\nFinished generating factoids based on topic: {SEED_METADATA_TOPICS[ti]}')
    else:
        print(f'Generating factoids based on topic: {SEED_METADATA_TOPICS[args.topic_index]}')
        factoid_gen.generate_factoids(args.topic_index)
        print(f'\n\nFinished generating factoids based on topic: {SEED_METADATA_TOPICS[args.topic_index]}')

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
