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
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
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
    "Financial Strategy",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
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
        if self.model_name.startswith("Qwen"):
            self.llm = LLM(model=self.model_name,
                quantization = "gptq",
                download_dir = os.environ['HF_HOME'],
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        else:
            self.llm = LLM(model=self.model_name,
                dtype="float16",
                download_dir = os.environ['HF_HOME'],
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "llama"
        
    def __is_valid_sentence(self, sentence):
    
        sentence = sentence.strip()
        
        if re.search(r'[\u4e00-\u9fff]', sentence):  # Unicode range for Chinese characters
            return False
        
        if len(sentence) > 10 and " " not in sentence:  # remove strings with no spaces
            return False
        
        if re.search(r'_[a-zA-Z]', sentence):  # Detect underscores replacing spaces
            return False
        
        if len(sentence.split(" ")) > 50 or len(sentence.split(" ")) < 5:
            return False
        
        return True


    def __extract_numbered_bullets(self, text, add_newline=False):
        bullet_pattern = r"^\s*\d+[\.\)-]\s+"
        lines = text.split("\n")
        numbered_bullets = [re.sub(bullet_pattern, "", line).strip() + ("\n" if add_newline else "")
                            for line in lines if re.match(bullet_pattern, line)]
        numbered_bullets = [s for s in numbered_bullets if self.__is_valid_sentence(s)]
        return numbered_bullets
    
    def __execute_LLM_task(self, prompt, max_new_tokens, temperature = 0.3, top_p = 0.9):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir = os.environ['HF_HOME'])
        tokenizer.lang = "en"
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that given a task returns a response following the exact structured output format specified in the prompt. Respond only in English"},
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
            stop=["===", "End of response."]
        )
        #model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
        outputs = self.llm.generate(text, sampling_params)
        response_text = tokenizer.decode(
            outputs[0].outputs[0].token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # Ensures proper spacing
        )
        #return outputs[0].outputs[0].text if outputs else ""
        return response_text

    def __score_chunks_by_topic(self, chunks, topic_threshold = np.ones(len(SEED_METADATA_TOPICS)) * 3):
        instruction_prompt = f"""
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
        - reasoning: "<short and concise explanation of relevance score>"

        ### Input:
        """

        relevant_chunks = []
        chunk_scores = []
        for ci, chunk in enumerate(chunks):
            relevance_scores = np.ones(len(SEED_METADATA_TOPICS))
            for ti,topic in enumerate(SEED_METADATA_TOPICS):
                prompt = instruction_prompt + f"\n- Topic: {topic}" + f"\n- Text: {chunk}"
                response = self.__execute_LLM_task(prompt, max_new_tokens=50).strip()
                #print('response: ', response)
                rsi = response.index("relevance: ")
                rei = response.index("\n")
                rel_score_str = response[rsi+11:rei]
                rel_score_str = re.findall(r'\d+', rel_score_str)
                if len(rel_score_str) > 0:
                    rel_score_str = rel_score_str[0]
                    rel_score = int(rel_score_str)
                else:
                    rel_score = 1
                relevance_scores[ti] = rel_score
            chunk_scores.append({
                'chunk_index': ci,
                'topics': SEED_METADATA_TOPICS,
                'relevance': relevance_scores.tolist()
            })
            if np.all((relevance_scores >= topic_threshold)==True):
                relevant_chunks.append(chunk)
                
        return relevant_chunks, chunk_scores

    def __generate_factoids(self, chunk_data):
        instruction_prompt = f"""
        ### Task:
        Given a text, extract verifiable factoids from it. A factoid is a **concise, factual statement** derived directly from the text.
        Generation guidelines:
        - **No opinions, adjectives or elaborations**
        - No extra details
        - Each factoid must be standalone, verifiable statement
        - Keep each factoid under 40 tokens.

        ### Input Format:
        - Text: <text>

        ### Output Format:
        Factoids:
        1. <factoid 1>
        2. <factoid 2>

        ### Now process this text:
        """


        instruction_prompt = instruction_prompt + f"\nText: {chunk_data}"

        summary = self.__execute_LLM_task(prompt=instruction_prompt, max_new_tokens=600, temperature=0.2, top_p = 0.8)
        if "Factoids:" in summary:
            fi = summary.index("Factoids:")
            summary_facts = summary[fi:]
            print('Generated Factoids:\n', summary_facts)
            return self.__extract_numbered_bullets(summary_facts, add_newline=True)
        return []


    def __generate_queries(self, fact_doc_text):

        instruction_prompt = f"""
        ### Task:
        Given a set of factoids, generate a query using one or multiple factoids. For each query, also generate the corresponding answer,
        the groundings and the reasonings. The groundings are the set/list of factoids used to answer the query and the reasoning are explanation of
        why or how the groundings answer the question.
        Generation guidelines:
        - Generate concise answers, summarized from the factoids listed in the groundings.
        - Groundings should consist of indices of factoids supporting the answer, directly picked from the input.
        - Reasoning should be an explanation of how each of the groundings support the answer
        - Keep reasoning under 100 tokens without adding any extra unnecessary adjectives, symbols or elaborations.
;
        ### Input format:
        ### Facts:
        1. <factoid 1>
        2. <factoid 2>
        ...

        ### Output format:
        Query: <question generated from fact(s) in the given text document>
        Answer: <answer to the question>
        Groundings: [<indices of factoids that support the answer>]
        Reasoning: <explanation of why/how the groundings support the answer>

        ### Now process this text:
        """

        instruction_prompt = instruction_prompt + f"\nFacts:\n{fact_doc_text}"

        summary = self.__execute_LLM_task(instruction_prompt, max_new_tokens=350)
        print('Generated Query:\n', summary)
        summary = f'Topic: {SEED_METADATA_TOPICS[self.topic_index]}\n{summary}'
        query_folder = f'data/queries/{self.model_folder}'
        query_files = [f for f in os.listdir(query_folder) if ".txt" in f]
        qn = len(query_files)
        with open(f'{query_folder}/query-example-{qn+1}.txt', 'w') as fp:
            fp.write(summary)

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
            self.chunks = [self.all_chunks[ci] for ci in range(len(self.scored_chunks)) if self.scored_chunks[ci]["relevance"][self.topic_index] >= 3.0]
            print('Filtered chunks: ', len(self.chunks))

            global_chunk_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/global_chunk_store.json'
            with open(global_chunk_fp, 'r+') as fp:
                chunks_obj = json.load(fp)
            chunk_arr = chunks_obj["chunks"]
            sel_chunk = [(pi,p) for pi,p in enumerate(chunk_arr) if p["chunk_filename"] == chunk_fn]
            if len(sel_chunk) == 1:
                fil_chunk = sel_chunk[0][1]
                seed_topics = fil_chunk["topics"]
                if SEED_METADATA_TOPICS[self.topic_index] not in seed_topics:
                    seed_topics.append(SEED_METADATA_TOPICS[self.topic_index])
                ex_factoids = fil_chunk["factoids"]
            else:
                seed_topics = [SEED_METADATA_TOPICS[self.topic_index]]
                ex_factoids = []
                fil_chunk = {
                    'chunk_filename': chunk_fn,
                    'factoids': ex_factoids,
                    'topics': seed_topics,
                    'sub_topics': []
                }
            for ci, text in enumerate(self.chunks):
                print(f'\nProcessing chunk {ci}...')
                generated_factoids = self.__generate_factoids(chunk_data=text)
                print('len of generated facts', len(generated_factoids))
                all_resp.extend(generated_factoids)
            ex_factoids.extend(all_resp)
            print('gen facts len', len(ex_factoids))
            fil_chunk['factoids'] = ex_factoids
            fil_chunk['topics'] = seed_topics
            if len(sel_chunk) == 1:
                chunk_arr[sel_chunk[0][0]] = fil_chunk
            else:
                chunk_arr.append(fil_chunk)
            chunks_obj["chunks"] = chunk_arr
            with open(global_chunk_fp, 'w') as fp:
                json.dump(chunks_obj, fp)
        elif instruction_type == INSTRUCTION_TYPES['QUERIES']:
            chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/global_chunk_store.json'
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_obj = [chunk for chunk in chunk_store["chunks"] if chunk["chunk_filename"] == chunk_fn]
            if len(chunk_obj) == 1:
                chunk_obj = chunk_obj[0]
            else:
                print('Chunk object not found!')
                SystemExit()
            print('length of the facts array: ', len(chunk_obj["factoids"]))
            chunk_factoids = chunk_obj["factoids"][:50]
            factoid_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(chunk_factoids))
            self.__generate_queries(factoid_str)
        elif instruction_type == INSTRUCTION_TYPES['CHUNK_CLASSIFICATION']:
            # Load chunked data
            chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_fp = f'data/chunked_data/{chunk_fn}.json'

            with open(chunk_fp, 'r') as fp:
                self.all_chunks = json.load(fp)['chunks']
            _, chunk_scores = self.__score_chunks_by_topic(chunks=self.all_chunks)
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
    #torch.distributed.destroy_process_group()
