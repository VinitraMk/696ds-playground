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
import ast
from awq import AutoAWQForCausalLM
import random

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
MAX_FACTOIDS_TO_SAMPLE = 50

'''
os.environ['VLLM_CACHE_ROOT'] = HF_CACHE_DIR
os.environ['VLLM_CACHE_DIR'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
'''

class BottomUpQueryGenerator:
    def __init__(self, filename, model_index=0,
            topic_index = 2):
        self.filename = filename
        self.company_abbr = filename.split('_')[1]
        self.page_data = {}
        self.model_index = model_index
        self.topic_index = topic_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Device enabled: {self.device}')

        # Load the model using vLLM
        self.model_name = MODELS[self.model_index]
        print('Model used: ', self.model_name)

        print('Model home: ', os.environ['HF_HOME'])
        if self.model_name.startswith("Qwen"):
            self.llm = LLM(model=f"./models/{self.model_name}",
                #quantization = "gptq",
                quantization = "awq",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                #gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        else:
            if "70b" in self.model_name:
                self.__quantize_llama(self.model_name)
                self.llm = LLM(model=f"{self.model_name}",
                    quantization="awq",
                    gpu_memory_utilization=0.8,
                    download_dir = HF_CACHE_DIR,
                    tensor_parallel_size=torch.cuda.device_count())
            else:
                self.llm = LLM(model=self.model_name,
                    download_dir = HF_CACHE_DIR,
                    tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "llama"

    def __quantize_llama(self, model_name):
        quant_model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            awq_config="4bit",
            cache_dir = HF_CACHE_DIR
        )
        quant_model.save_pretrained(f"{HF_CACHE_DIR}/{model_name}-awq")

    def __is_valid_sentence(self, sentence):
    
        sentence = sentence.strip()
        
        if re.search(r'[\u4e00-\u9fff]', sentence):  # Unicode range for Chinese characters
            return False
        
        if len(sentence) > 10 and " " not in sentence:  # remove strings with no spaces
            return False
        
        if re.search(r'_[a-zA-Z]', sentence):  # Detect underscores replacing spaces
            return False
        
        if len(sentence.split(" ")) > 100 or len(sentence.split(" ")) < 7:
            return False
        
        return True

    def __extract_factoid_list(self, text):
        match = re.search(r'Factoids:\s*(\[[\s\S]*?\])', text)
    
        if match:
            list_str = match.group(1)  # Extract the list as a string
            try:
                factoids = ast.literal_eval(list_str)  # Safely evaluate string as Python list
                if isinstance(factoids, list):
                    return factoids
            except Exception as e:
                print("Failed to parse factoid list:", e)
        return []

    def __extract_and_clean_factoids(self, text, add_newline=False):
        factoid_list = self.__extract_factoid_list(text)
        
        clean_factoids = [s for s in factoid_list if self.__is_valid_sentence(s)]
        return clean_factoids

    def __extract_json_from_text(self, text):
        # Attempt to extract top-level key-value pairs with regex
        pattern = r'"(\w+)":\s*(\[[\s\S]*?\]|\{[\s\S]*?\}|"(?:\\.|[^"\\])*"|[^,}\n\r]+)'
        
        matches = re.findall(pattern, text)
        result = {}
        valid_key_vals = [kv for kv in matches if (kv[0] == "query" or kv[0] == "answer" or kv[0] == "reasonings" or kv[0] == "groundings")]

        for key, raw_value in valid_key_vals:
            # Try parsing the value
            try:
                # Clean stray trailing commas or whitespace
                value = raw_value.strip().rstrip(',')

                # Handle raw JSON strings, arrays, or objects
                if value.startswith('"') and value.endswith('"'):
                    value = json.loads(value)
                elif value.startswith('[') or value.startswith('{'):
                    value = json.loads(value)
                else:
                    value = value.strip('"')
                
                result[key] = value
            except Exception as e:
                print(f"Skipping key '{key}' due to parsing error: {e}")
        print('json object result', result)
        try:
            json.loads(str(result))
            return result
        except ValueError:
            return {}

    
    def __execute_LLM_task(self, prompt, max_new_tokens, temperature = 0.3, top_p = 0.9):
        #tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir = os.environ['HF_HOME'])
        tokenizer = AutoTokenizer.from_pretrained(f"./models/{self.model_name}")
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
            stop=["###EOF###"]
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

    def __score_chunks_by_topic(self, chunks, topic_threshold = np.ones(len(SEED_METADATA_TOPICS)) * RELEVANCE_THRESHOLD):
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
                if "relevance: " in response:
                    rsi = response.index("relevance: ")
                    rei = response.index("\n")
                    rel_score_str = response[rsi+11:rei]
                    rel_score_str = re.findall(r'\d+', rel_score_str)
                    if len(rel_score_str) > 0:
                        rel_score_str = rel_score_str[0]
                        rel_score = int(rel_score_str)
                    else:
                        rel_score = 1
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
    
    def __cleanup_factoids(self, factoids):
        pre_instruction_prompt = f"""
        ### Task:
        Given an input text, return a text that is cleaned (i.e remove incorrect or unnecessary punctuations) and grammatically correct, if the input text contains a meaningful sentence.
        If the text does not contain a valid or meaningful sentence (e.g., it's just fragments, broken syntax, or random characters), return:
        NOT A SENTENCE

        ### Input Format:
        - Text: <line of text>

        ### Output Format:
        Response: <cleaned sentence> / ### NOT A SENTENCE ###

        ### Example Input/Output
        Input:
        Text: "he go to market yesterday"
        Output:
        Response: "He went to the market yesterday."

        Input:
        Text: "%% weird fragment 21 and not finished"
        Output:
        Response: NOT A SENTENCE

        ### Now process this text:
        """

        clean_factoids = []
        for factoid in factoids:
            instruction_prompt = pre_instruction_prompt + f"\nText: {factoid}"
            summary = self.__execute_LLM_task(prompt=instruction_prompt, max_new_tokens=200, temperature=0.1, top_p = 0.8)
            print('cleaned factoid: ', summary)
            if ("Response: " in summary) and ("\n" in summary) and ("NOT A SENTENCE" not in summary):
                ri = summary.index("Response: ")
                nli = summary.index("\n")
                cleaned_text = summary[(ri+10):nli]
                clean_factoids.append(cleaned_text)
        return clean_factoids


    def __generate_factoids(self, chunk_data):
        instruction_prompt = f"""
        ### Task:
        Given a text, extract verifiable factoids from it. A factoid is a **factual statement** derived directly from the text.

        ### Generation guidelines:
        - **No opinions, adjectives, elaborations or extra details**
        - Each factoid must be standalone, verifiable statement
        - Keep each factoid under 100 tokens.
        - Focus only on information related to the provided topic.
        - If no factoids present relevant to the provided topic, return response saying "NO FACTOIDS PRESENT"
        - If factoids are found, present the output stating "Factoids: <list of factoids>".

        ### Input Format:
        - Text: <text>

        ### Output Format:
        Factoids: ["Factoid 1", "Factoid 2"]
        
        ### Now process this text:
        """


        instruction_prompt = instruction_prompt + f"\nText: {chunk_data}"

        summary = self.__execute_LLM_task(prompt=instruction_prompt, max_new_tokens=500, temperature=0.2, top_p = 0.8)
        print('Generated response', summary)
        if "Factoids:" in summary:
            fi = summary.index("Factoids:")
            summary_facts = summary[fi:]
            print('Generated Factoids:\n', summary_facts)
            #factoid_bullets = self.__extract_numbered_bullets(summary_facts, add_newline=True)
            return self.__extract_and_clean_factoids(summary_facts)
        return []
    
    def __is_valid_query_item(self, query_dict):
        if ("query" not in query_dict) or ("answer" not in query_dict) or ("groundings" not in query_dict) or ("reasoning" not in query_dict):
            return False
        if len(query_dict["query"].split(" ")) > 200:
            return False
        elif len(query_dict["answer"].split(" ")) > 250:
            return False
        elif len(query_dict["reasoning"].split(" ")) > 200:
            return False
        return True

    def __extract_clean_queryset(self, gen_query_response):
        instruction_prompt = """
        ### Task:
        Extract and format the following sections from the provided text into a **strict Markdown structure**:

        ### Required Sections:  
        - **Query**: The question posed (beginning with "Query:").  
        - **Answer**: The response bullet points (beginning with "ANSWER:").  

        ### Generation guidelines
        - Output **only** the desired markdown structure, no explanations.

        ### Formatting Rules:  
        - Clean all text, remove extra spaces, unnecessary or incorrect punctuations.
        - Remove trailing spaces, commas, hyphens.
        - Fix all obvious typos
        - Correct grammatical errors.

        ### Input Format:
        **Query:** <query text>
        **Answer:** <answer text>

        ### Output Format:
        {
            "query": <query text>,
            "answer": <answer text>,
        }

        ### Example Input:  
        **Query:** Why adjust_[inventory values=_ quarterly?  
        **ANSWER:**  
        - Rapid innovation reduces useful lifetimes…  

        ### Example Output: 
        {
            "query": "Why adjust inventory values quarterly?",
            "answer": "Rapid innovation reduces useful lifetimes.",
        } 

        ### Task Input
        """

        instruction_prompt = instruction_prompt + f"\n**Text:** {gen_query_response}"
        summary = self.__execute_LLM_task(instruction_prompt, max_new_tokens=3000, temperature=0.1, top_p = 0.9)
        print('\nGenerated structured Query:\n', summary)
        query_dict = self.__extract_json_from_text(summary)
        if query_dict != {}:
            print('Query dict', query_dict)
            with open(f'./data/queries/{self.model_folder}/{self.filename}_gen_queries.json', 'r') as fp:
                queries = json.load(fp)
            topic_queries = [tq for tq in queries["queries"] if tq["topic"] == SEED_METADATA_TOPICS[self.topic_index]]
            print("topic queries", topic_queries, len(topic_queries), query_dict)
            if len(topic_queries) > 0:
                topic_queries = topic_queries[0]
                topic_queries["query_sets"].append(query_dict)
                for tq in queries["queries"]:
                    if tq["topic"] == SEED_METADATA_TOPICS[self.topic_index]:
                        tq["query_sets"] = topic_queries
            else:
                topic_queries = { "topic": SEED_METADATA_TOPICS[self.topic_index], "query_sets": [] }
                topic_queries["query_sets"].append(query_dict)
                queries["queries"].append(topic_queries)

            with open(f'./data/queries/{self.model_folder}/{self.filename}_gen_queries.json', 'w') as fp:
                json.dump(queries, fp)


    def __generate_query(self, fact_doc_text, metadata):

        instruction_prompt = f"""
        ### Task:
        Analyze the provided set of factoids and the metadata and generate **only one structured response** as described below.

        ### Desired response structure:  
        - **Query**: Write a complex question based on the factoids.
        - **Answer**: Write a concise answer to the question generated.

        ### Generation Rules
        - **Do not use chinese characters** in your response.
        - Phrase your response as concisely as possible.
        - Keep the query under 100 tokens while the answer under 200 tokens.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - End the response with `###EOF###`.  
        - Use the example structure to return the final response.
        - Label your chain-of-thought as `###COT###` and the final answer as `###FINALRESPONSE###`.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        **Metadata:** <meta data of the main company upon which the factoids are based.>
        **Factoids:** [\<list of factoids\>]

        ### Output format:
        **Query:** <question generated from fact(s) in the given text document>
        **Answer:** <answer to the question>

        ### Example Output:

        ###FINAL RESPONSE###
        **Query:** How does Apple’s commitment to achieving carbon neutrality across its supply chain and
        products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships,
        and long-term profitability, and what are the potential risks and rewards associated with
        this aggressive ESG strategy?

        **Answer:**
        Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
        On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation.
        From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments.

        ### Input for your task:
        """

        instruction_prompt = instruction_prompt + f"\n**Metadata:** {metadata}\n**Factoids:**{fact_doc_text}"

        summary = self.__execute_LLM_task(instruction_prompt, max_new_tokens=1500, temperature=0.3, top_p = 0.5)
        print('Generated Response:\n', summary)
        #self.__cleanup_query(summary)
        self.__extract_clean_queryset(summary)


    def run(self, instruction_type=INSTRUCTION_TYPES['FACTOIDS']):
        st = time()

        log_fp = f'logs/bu-query-logs.txt'
        log_file = open(log_fp, 'w')
        old_stdout = sys.stdout
        sys.stdout = log_file

        # Filter relevant chunks

        all_resp = []
        if instruction_type == INSTRUCTION_TYPES['FACTOIDS']:
            #chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_fn = f'{self.filename}_chunked'
            chunk_fp = f'data/chunked_data/{chunk_fn}.json'
            scored_chunk_fp = f'data/chunked_data/scored_chunks/{chunk_fn}.json'
            with open(chunk_fp, 'r') as fp:
                self.all_chunks = json.load(fp)["chunks"]
            with open(scored_chunk_fp, 'r') as fp:
                self.scored_chunks = json.load(fp)
            self.chunks = [{ 'chunk_index': ci, 'text': self.all_chunks[ci] } for ci in range(len(self.scored_chunks)) if self.scored_chunks[ci]["relevance"][self.topic_index] >= RELEVANCE_THRESHOLD]
            print('Filtered chunks: ', len(self.chunks))

            file_chunkstore_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
            if os.path.exists(file_chunkstore_fp):
                with open(file_chunkstore_fp, 'r+') as fp:
                    chunks_obj = json.load(fp)
                chunk_arr = chunks_obj["chunks"] 
            else:
                chunks_obj = { "chunks": [] }
                chunk_arr = []
            
            for ci in range(len(self.all_chunks)):
                chunk_dict = [chunk for chunk in self.chunks if chunk["chunk_index"] == ci]
                if len(chunk_dict) > 0:
                    chunk_dict = chunk_dict[0]
                    print(f'\nProcessing chunk {ci}...')
                    generated_factoids = self.__generate_factoids(chunk_data=chunk_dict["text"])
                    generated_factoids_objarr = [{ 'topic': SEED_METADATA_TOPICS[self.topic_index], 'factoid': factoid } for factoid in generated_factoids]
                    print('len of generated facts', len(generated_factoids))
                    if len(chunk_arr) > 0:
                        existing_topics = chunk_arr[chunk_dict["chunk_index"]]["topics"]
                        if (len(generated_factoids) > 0) and (SEED_METADATA_TOPICS[self.topic_index] not in existing_topics):
                            existing_topics.append(SEED_METADATA_TOPICS[self.topic_index])
                        existing_factoids = chunk_arr[chunk_dict["chunk_index"]]["factoids"]
                    else:
                        existing_topics = [SEED_METADATA_TOPICS[self.topic_index]] if len(generated_factoids) > 0 else []
                        existing_factoids = []
                    existing_factoids.extend(generated_factoids_objarr)
                    chunk_resp = {
                        'chunk_index': chunk_dict["chunk_index"],
                        'factoids': existing_factoids,
                        'topics': existing_topics,
                    }
                    all_resp.append(chunk_resp)
                else:
                    chunk_resp = {
                        'chunk_index': ci,
                        'factoids': [],
                        'topics': []
                    }
                    all_resp.append(chunk_resp)
                
            chunks_obj["chunks"] = all_resp
            if os.path.exists(file_chunkstore_fp):
                with open(file_chunkstore_fp, 'w') as fp:
                    json.dump(chunks_obj, fp)
            else:
                fp = open(file_chunkstore_fp, 'x')
                json.dump(chunks_obj, fp)
        elif instruction_type == INSTRUCTION_TYPES['QUERIES']:
            chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
            
            #chunk_obj = [chunk for chunk in chunk_store["chunks"] if chunk["chunk_filename"] == chunk_fn]
            if os.path.exists(chunk_store_fp):
                with open(chunk_store_fp, 'r') as fp:
                    chunk_store = json.load(fp)
                chunk_arr = chunk_store["chunks"]
                all_factoids = []
                for chunk in chunk_arr:
                    if len(chunk["factoids"]) > 0:
                        all_factoids.extend(chunk["factoids"])
                print('length of the facts array: ', len(all_factoids))
                #chunk_topics = ",".join(chunk_obj["topics"])
                #random_indices = random.sample(range(0, len(all_factoids)), MAX_FACTOIDS_TO_SAMPLE)
                filtered_factoids = [factoid for factoid in all_factoids if factoid["topic"] == SEED_METADATA_TOPICS[self.topic_index]]
                #factoid_subarr = all_factoids[:MAX_FACTOIDS_TO_SAMPLE]
                factoid_subarr = filtered_factoids[:MAX_FACTOIDS_TO_SAMPLE]
                metadata = f'Company: {self.company_abbr} | SEC Filing: 10-K'
                #for idx in random_indices:
                    #factoid_subarr.append(all_factoids[idx])
                factoid_str = "[" + ",".join(f"{item['factoid']}" for item in factoid_subarr) + "]"
                self.__generate_query(factoid_str, metadata)
            else:
                print('Chunk store not found!')
                SystemExit()
            
        elif instruction_type == INSTRUCTION_TYPES['CHUNK_CLASSIFICATION']:
            # Load chunked data
            #chunk_fn = '10-K_AMD_20231230_chunked'
            chunk_fn = f'{self.filename}_chunked'
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
    parser.add_argument('--instruction_type', type=str, default=INSTRUCTION_TYPES['FACTOIDS'], required=False)
    parser.add_argument('--topic_index', type=int, default = 0, required = False)

    args = parser.parse_args()

    filename = '10-K_AMD_20231230'
    bu_query_gen = BottomUpQueryGenerator(
        filename=filename, model_index=args.model_index,
        topic_index=args.topic_index
    )
    bu_query_gen.run(instruction_type=args.instruction_type)
    #torch.distributed.destroy_process_group()
