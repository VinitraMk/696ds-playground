import argparse
import json
import os
import re
import sys
import torch
from time import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing
import numpy as np

MODELS = [
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "meta-llama/Meta-Llama-3-70B"
]

INSTRUCTION_TYPES = {
    'FACTOIDS': 'factoids',
    'QUERIES': 'queries',
    'METADATA': 'metadata',
    'CHUNK_CLASSIFICATION': 'chunk_classification'
}

SEED_METADATA_TOPICS = [
    "Financial Strategy",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
    "Risk Factors and Challenges",
    "Market Trends, Economic Environment, and Industry Dynamics"
]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR


class BottomUpQueryGenerator:
    def __init__(self, filename, model_index=0,
            topic_index = 2,
            min_factoids=5, max_factoids=20):
        self.filename = filename
        self.page_data = {}
        self.model_index = model_index
        self.min_factoids = min_factoids
        self.max_factoids = max_factoids
        self.topic_index = topic_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Device enabled: {self.device}')

        # Load the model using vLLM
        self.model_name = MODELS[self.model_index]
        print('Model used: ', self.model_name)
        self.llm = LLM(model=self.model_name,
            quantization = "gptq",
            download_dir = os.environ['HF_HOME'],
            tensor_parallel_size=torch.cuda.device_count())


    def __extract_numbered_bullets(self, text, add_newline=False):
        bullet_pattern = r"^\s*\d+[\.\)-]\s+"
        lines = text.split("\n")
        numbered_bullets = [re.sub(bullet_pattern, "", line).strip() + ("\n" if add_newline else "")
                            for line in lines if re.match(bullet_pattern, line)]
        return numbered_bullets
    
    def __execute_LLM_task(self, prompt, max_new_tokens, temperature = 0.3, top_p = 0.9):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir = os.environ['HF_HOME'])
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.5,
            stop=["<STOP>"]
        )
        outputs = self.llm.generate(text, sampling_params)
        return outputs[0].outputs[0].text if outputs else ""

    def __filter_relevant_chunks(self, chunks, metadata_topics=["Financial Strategy"], topic_threshold = np.array([3])):
        instruction_prompt = f"""
        ### Instruction:
        You are an AI assistant that given a text, returns a response following the exact structured output format specified in the prompt.

        ### Task:
        Given a chunk of text and a topic, determine if the text is relevant to the topic.
        Respond with a score on a scale from 1 to 5, rating the relevance of the text to the topic.
        Consider the following rubric to rate the relevance of the text:
        1 = Completely irrelevant i.e content having no information about the topic
        2 = Slightly relevant i.e content vaguely mentioning the topic
        3 = Moderately relevant i.e content having some information about the topic
        4 = Highly relevant i.e content having good amount of information about the topic
        5 = Perfectly relevant i.e content being primarily about the topic provided

        ### Input format:
        - Topic
        - Text

        ### Output format:
        - relevance: "<score in scale 1-5>"
        - reasoning: "<short explanation of relevance score>"

        ### Input:
        """

        relevant_chunks = []
        chunk_scores = []
        for ci, chunk in enumerate(chunks):
            relevance_scores = np.zeros(len(metadata_topics))
            for ti,topic in enumerate(metadata_topics):
                prompt = instruction_prompt + f"\n- Topic: {topic}" + f"\n- Text: {chunk}"
                response = self.__execute_LLM_task(prompt, max_new_tokens=100).strip()
                rsi = response.index("relevance: ")
                rei = response.index("\n")
                rel_score = int(response[rsi+11:rei].replace('"',''))
                relevance_scores[ti] = rel_score
            chunk_scores.append({
                'chunk_index': ci,
                'topics': metadata_topics,
                'relevance': relevance_scores.tolist()
            })
            if np.all((relevance_scores >= topic_threshold)==True):
                relevant_chunks.append(chunk)
                
        return relevant_chunks, chunk_scores

    def __generate_factoids(self, chunk_data):
        instruction_prompt = f"""
        ### Instruction:
        You are an AI assistant that given a text, returns a response following the exact structured output format specified in the prompt.

        ### Task:
        Given a chunk of text, generate a set facts from it.

        ### Input format:
        - Text: <text>

        ### Output format:
        ### Facts:
        1. <fact 1>
        2. <fact 2>
        ...

        ### Input:
        """

        '''
        instruction_prompt = """
        ### Instruction:
        You are an AI assistant that given a text, returns a response following the exact structured output format specified in the prompt.

        ### Task:
        Given a chunk of text, generate a set factoids from it. Each factoid should represent a standalone piece of factual information derived directly from the text.

        ### Input format:
        - Text: <text>
        """
        '''

        instruction_prompt = instruction_prompt + f"\nText: {chunk_data}"

        summary = self.__execute_LLM_task(prompt=instruction_prompt, max_new_tokens=500, temperature=0.5)
        fi = summary.index("### Facts:")
        summary_facts = summary[fi:]
        print('Generated Factoids:\n', summary_facts)
        return self.__extract_numbered_bullets(summary_facts, add_newline=True)

    def __generate_queries(self, document_text):
        instruction = (
            f"### INSTRUCTION:\nGenerate 5 multi-factoid queries answerable using the given text.\n"
            "For each query, provide:\n"
            "- The query itself.\n"
            "- An answer derived from factoids in the text.\n"
            "- A list of grounding factoids (by index).\n"
            "- Reasoning explaining the answer.\n\n"
            "### TEXT:\n"
            f"{document_text}\n\n"
            "### QUERIES:\n1."
        )

        summary = self.__execute_LLM_task(instruction, max_new_tokens=1024)
        print('Generated Queries:\n', summary)

    def run(self, instruction_type=INSTRUCTION_TYPES['FACTOIDS']):
        st = time()

        log_fp = f'logs/bu-query-logs.txt'
        log_file = open(log_fp, 'w')
        old_stdout = sys.stdout
        sys.stdout = log_file

        # Filter relevant chunks

        all_resp = []
        if instruction_type == INSTRUCTION_TYPES['FACTOIDS']:
            chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_fp = f'data/chunked_data/{chunk_fn}.json'
            scored_chunk_fp = f'data/chunked_data/scored_chunks/{chunk_fn}.json'
            with open(chunk_fp, 'r') as fp:
                self.all_chunks = json.load(fp)["chunks"]
            with open(scored_chunk_fp, 'r') as fp:
                self.scored_chunks = json.load(fp)
            self.chunks = [self.all_chunks[ci] for ci in range(len(self.scored_chunks)) if self.scored_chunks[ci]["relevance"][self.topic_index] >= 3]
            print('Filtered chunks: ', len(self.chunks))

            '''
            for ci, text in enumerate(self.chunks):
                print(f'\nProcessing chunk {ci}...')
                generated_factoids = self.__generate_factoids(chunk_data=text)
                all_resp.extend(generated_factoids)

            txt_file = f'data/factoids/factoids-{chunk_fn.replace("_chunked", "")}.txt'
            all_resp.insert(0, '### FACTOIDS:\n')
            with open(txt_file, 'w') as fp:
                fp.writelines(all_resp)
            '''
            with open(f'data/chunked_data/global_chunk_store/global_chunk_store.json', 'r+') as fp:
                chunks_obj = json.load(fp)
            chunk_arr = chunks_obj["chunks"]
            sel_chunk = [(pi,p) for pi,p in enumerate(chunk_arr) if p["chunk_filename"] == chunk_fn]
            if len(sel_chunk) == 1:
                fil_chunk = sel_chunk[0][1]
            else:
                fil_chunk = {
                    'chunk_filename': chunk_fn,
                    'factoids': [],
                    'topics': [SEED_METADATA_TOPICS[self.topic_index]],
                    'sub_topics': []
                }
            for ci, text in enumerate(self.chunks):
                print(f'\nProcessing chunk {ci}...')
                generated_factoids = self.__generate_factoids(chunk_data=text)
                all_resp.extend(generated_factoids)
            fil_chunk['factoids'] = all_resp
            fil_chunk['topics'] = [SEED_METADATA_TOPICS[self.topic_index]]
            if len(sel_chunk) == 1:
                chunk_arr[sel_chunk[0][0]] = fil_chunk
            else:
                chunk_arr.append(fil_chunk)
            chunks_obj["chunks"] = chunk_arr
            with open(f'data/chunked_data/global_chunk_store/global_chunk_store.json', 'w') as fp:
                json.dump(chunks_obj, fp)
        elif instruction_type == INSTRUCTION_TYPES['QUERIES']:
            chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_store_fp = 'data/chunked_data/global_chunk_store/global_chunk_store.json'
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_obj = [chunk for chunk in chunk_store["chunks"] if chunk["chunk_filename"] == chunk_fn]
            if len(chunks_obj) == 1:
                chunks_obj = chunks_obj[0]
            else:
                print('Chunk object not found!')
                SystemExit()
            chunk_factoids = chunk_obj["factoids"]
            factoid_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(chunk_factoids))
            self.__generate_queries(factoid_str)
        elif instruction_type == INSTRUCTION_TYPES['CHUNK_CLASSIFICATION']:
            # Load chunked data
            chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_fp = f'data/chunked_data/{chunk_fn}.json'

            with open(chunk_fp, 'r') as fp:
                self.all_chunks = json.load(fp)['chunks']
            _, chunk_scores = self.__filter_relevant_chunks(chunks=self.all_chunks, metadata_topics=[SEED_METADATA_TOPICS[self.topic_index]])
            #print('Filtered chunks: ', len(self.chunks))
            scored_chunk_fp = f'data/chunked_data/scored_chunks/{chunk_fn}.json'
            with open(scored_chunk_fp, 'w') as fp:
                json.dump(chunk_scores, fp)

            
        print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
        sys.stdout = old_stdout
        log_file.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default=5, required=False)
    parser.add_argument('--min_factoids', type=int, default=5, required=False)
    parser.add_argument('--max_factoids', type=int, default=10, required=False)
    parser.add_argument('--instruction_type', type=str, default=INSTRUCTION_TYPES['FACTOIDS'], required=False)
    parser.add_argument('--topic_index', type=int, default = 0, required = False)

    args = parser.parse_args()

    filename = '000000248824000012-amd-20231230'
    bu_query_gen = BottomUpQueryGenerator(
        filename=filename, model_index=args.model_index,
        topic_index=args.topic_index,
        min_factoids=args.min_factoids, max_factoids=args.max_factoids
    )
    bu_query_gen.run(instruction_type=args.instruction_type)
    torch.distributed.destroy_process_group()
