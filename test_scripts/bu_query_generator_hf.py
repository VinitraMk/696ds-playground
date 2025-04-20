import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from time import time
import torch
import re
import os
from transformers import BitsAndBytesConfig
import sys
from modelscope import snapshot_download
import json

MODELS = [
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "qwen/Qwen-32B",
    "meta-llama/Meta-Llama-3-70B"
]

INSTRUCTION_TYPES = {
    'FACTOIDS': 'factoids',
    'QUERIES': 'queries',
    'METADATA': 'metadata'
}

SEED_METADATA_TOPICS = ["Leadership & Governance", "ESG & Sustainability", "Risk Factors", "Financial Strategy", "Business Growth & Strategy", "Technology & Innovation"]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR

class BottomUpQueryGenerator:

    def __init__(self, filename, model_index = 0, chunk_length = 500,
        max_tokens = 200, min_factoids = 5, max_factoids = 20):
        self.filename = filename
        self.page_data = {}
        self.model_index = model_index
        self.chunk_length = chunk_length
        self.max_tokens = max_tokens
        self.min_factoids = min_factoids
        self.max_factoids = max_factoids
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device enabled: ', self.device)

    def __extract_numbered_bullets(self, text, add_newline = False):
        bullet_pattern = r"^\s*\d+[\.\)-]\s+"

        # Extract matching lines
        lines = text.split("\n")
        if add_newline:
            numbered_bullets = [re.sub(bullet_pattern, "", line).strip() + "\n" for line in lines if re.match(bullet_pattern, line)]
        else:
            numbered_bullets = [re.sub(bullet_pattern, "", line).strip() for line in lines if re.match(bullet_pattern, line)]

        return numbered_bullets
    
    def __filter_relevant_chunks(self, chunks, metadata_topic = "Financial Strategy"):
        pre_instruction_prompt = """
        ### Task:
        Given a chunk of text and a topic, identify (answer yes/no) whether the text addresses or is relevant to the topic.

        ### Input format:
        - Topic
        - Text

        ### Output format:
        - yes/no 

        ### Input:
        """

        pre_instruction_prompt = pre_instruction_prompt + f"\n- Topic: {metadata_topic}"

        for chunk in chunks:
            instruction_prompt = pre_instruction_prompt + f"\n- Text: {chunk}"
            inputs = self.tokenizer(instruction_prompt, return_tensors="pt").to(self.device)
            # Generate summary
            #summary = summarizer(f"{instruction}: {text}", max_length=4096, do_sample=True)[0]["generated_text"]
            op = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                #eos_token_id = self.tokenizer.eos_token_id
            )
            summary = self.tokenizer.decode(op[0], skip_special_tokens=True)
            print('response for chunk relevance:', summary)
        return []

    
    def __generate_factoids(self, chunk_data, category = "Financial Strategy"):
        #instruction prompt
        #instruction = f'### INSTRUCTION: \nUsing the text given below, generate {self.min_factoids} factoids from it.' + \
        #' Separate these factoids by numbered bullets and present only the factoids in your response, add ###Factoids as the header before presenting them.'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        instruction = (
            "A factoid is a concise, factual statement that captures a key point from the text.\n"
            f"You are an AI assistant that summarizes factoids related to {category}, from a given text extracted from SEC 10K filing."
            "Present the generated factoids in a numbered list."
            "### TEXT:\n"
            f"{chunk_data}\n\n",
            "\n### FACTOIDS:\n"
            "1."
        )

        inputs = self.tokenizer(f"{instruction} - \n### TEXT: {chunk_data}", return_tensors="pt").to(self.device)
        # Generate summary
        #summary = summarizer(f"{instruction}: {text}", max_length=4096, do_sample=True)[0]["generated_text"]
        op = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            #eos_token_id = self.tokenizer.eos_token_id
        )
        summary = self.tokenizer.decode(op[0], skip_special_tokens=True)
        print('summary:', summary)
        generated_factoids = self.__extract_numbered_bullets(text = summary, add_newline=True)
        return generated_factoids
    
    def __generate_metadata(self, chunk_data):
        instruction = f'### INSTRUCTION: \nUsing the text given below, generate metadata of topics covered in the text.' + \
        ' Separate these metadata by numbered bullets and present only the metadata of topics in your response.'

        inputs = self.tokenizer(f"{instruction} - \n### TEXT: {chunk_data}", return_tensors="pt").to(self.device)

        # Generate summary
        #summary = summarizer(f"{instruction}: {text}", max_length=4096, do_sample=True)[0]["generated_text"]
        op = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=0.5,
            top_p=0.8,
            do_sample=True,
            eos_token_id = self.tokenizer.eos_token_id
        )
        summary = self.tokenizer.decode(op[0], skip_special_tokens=True)
        generated_metadata = self.__extract_numbered_bullets(summary)

        return generated_metadata

    def __cleanup_queries(self):
        query_fn = f'data/queries/queries-{self.filename}.txt'
        with open(query_fn, 'r') as fp:
            query_data = fp.readlines()
        gen_queries = []
        for ln in query_data:
            if ln.startswith('Query:'):
                print(ln)
                qi = ln.index('Query: ')
                query = ln[qi+7:].strip()
                q_dict = {
                    'query': query,
                    'groundings': [],
                    'reasoning': ''
                }
            elif ln.startswith('Groundings:'):
                gi = ln.index('Groundings: ')
                groundings = ln[gi+12:].strip().split(',')
                groundings = [int(s) for s in groundings]
                q_dict['groundings'] = groundings
            elif ln.startswith('Reasoning:'):
                ri = ln.index('Reasoning: ')
                reasoning = ln[ri+11:]
                q_dict['reasoning'] = reasoning
                gen_queries.append(q_dict)
        query_json_fn = f'data/queries/queries-{self.filename}.json'
        with open(query_json_fn, 'w') as fp:
            json.dump(gen_queries, fp)

    def __generate_query(self, document_text):
        instruction_prompt = (f'### INSTRUCTION: \nGenerate a set of 5 queries answerable using multiple factoids in the given text. ',
        ' For each query, generate the answer, groundings (set of factoids that answer the query and reasoning (explanation of why the groundings support the answer)',
        ' Present the generated set as a numbered list.',
        ' \n### TEXT:\n',
        f'{document_text}')
        '''
        instruction_prompt = """
        ### Instruction
        You are an AI assistant trained to follow a task and generate the output of the task in a structued JSON format from a given text.
        
        ### Task
        Given a document consisting a list of factoids, generate a structured JSON list, consisting of:
        1. A **query** based on a factoid from the document.
        2. A **fact-based answer** derived from the factoid.
        3. The **groundings**, which are the indices of the factoids supporting the answer.
        4. A **reasoning statement** explaining how the factoid supports the answer.

        ### Input:
        You will receive a document text containing set of factoids.

        ### Output Format:
        Return a JSON array where each object has the following structure:
        [
            {
                "query": "<Question based on the document>",
                "answer": "<Concise factual answer from the document>",
                "groundings": [<List of factoid index or indices>],
                "reasoning": "<Brief explanation of how the factoids support the answer>"
            },
            ...
        ]

        ### Example output:
        [
            {
                "query": "What is the company's fiscal year end date?",
                "answer": "The company's fiscal year end date is December 30, 2023.",
                "groundings": [1],
                "reasoning": "Factoid 1 states that the company's fiscal year ended on December 30, 2023."
            },
            {
                "query": "Where is the company's headquarters located?",
                "answer": "The company's headquarters is in Cupertino, California.",
                "groundings": [2],
                "reasoning": "Factoid 2 explicitly states the headquarters location."
            }
        ]

        ### Document:
        """
        instruction_prompt = instruction_prompt + f"\n{document_text}"
        #document = f"""{factoid_data}"""
        #instruction_prompt = instruction_prompt.format(document_text=document_text)
        '''

        inputs = self.tokenizer(f"{instruction_prompt}", return_tensors = "pt").to(self.device)
        #inputs = self.tokenizer(instruction_prompt, return_tensors = "pt").to(self.device)
        op = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            eos_token_id = self.tokenizer.eos_token_id
        )
        summary = self.tokenizer.decode(op[0], skip_special_tokens=True)
        print('generated summary: ', summary)
        '''
        ti = summary.index('### Document:')
        qi = summary[ti:].index('### QUERIES:')
        summary_queries = summary[ti:][qi:]
        print('Queries:\n', summary_queries)
        with open(f'data/queries/queries-{self.filename}.txt', 'w') as fp:
            fp.write(summary_queries)
        '''

    def __cleanup_factoids(self, factoids):
        clean_factoids = []
        for ln in factoids:
            if "###Factoids###." in ln:
                ln.replace('###Factoids###.','')
            elif "###END OF RESPONSE" in ln:
                ssi = ln.index('###END OF RESPONSE')
                ln = ln[:ssi]
            elif ln.strip() == "":
                continue 
            clean_factoids.append(ln)
        return clean_factoids
        
    def run(self, instruction_type = INSTRUCTION_TYPES['FACTOIDS']):
        st = time()
        '''
        json_fp = f'base_dataset/SEC-10K/parsed_data/custom/parsed-{self.filename}.json'
        with open(json_fp, 'r') as fp:
            self.page_data = json.load(fp)
        '''

        chunk_fp = f'data/chunked_data/10-K_AMD_20231230_chunked.json'
        log_fp = f'logs/bu-query-logs.txt'
        with open(chunk_fp, 'r') as fp:
            self.all_chunks = json.load(fp)['chunks']
        log_file = open(log_fp, 'w')
        old_stdout = sys.stdout
        sys.stdout = log_file

        # Define the model name
        
        model_name = MODELS[self.model_index]
        if model_name == 'qwen/Qwen-32B':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )
            model_dir = snapshot_download(model_name)
            

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True, cache_dir = os.environ['HF_HOME'])
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map={"": self.device}, # forces model to use gpu
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                cache_dir = os.environ['HF_HOME'],
                trust_remote_code = True)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True
            )
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = os.environ['HF_HOME'])
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": self.device}, # forces model to use gpu
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                cache_dir = os.environ['HF_HOME'])

        # filter out relevant chunks
        self.chunks = self.__filter_relevant_chunks(self.all_chunks)

        all_resp = []
        all_topics = []
        all_queries = []
        if instruction_type == INSTRUCTION_TYPES['FACTOIDS']:
            ci = 0
            while ci < len(self.chunks):
                print(f'\nProcessing chunk: {ci}')
                #text = section_data[chunk_si:(chunk_si + self.chunk_length)]
                text = self.chunks[ci]
                print('conditional text: ', text)
                print('text length: ', len(text))
                generated_factoids = self.__generate_factoids(text, "Financial Strategy")
                all_resp = all_resp + generated_factoids
                ci+=1
            txt_file = f'data/factoids/factoids-{self.filename}.txt'
            print('all unclean factoids')
            print(all_resp)
            all_resp = self.__cleanup_factoids(all_resp)
            print('\n\nunclean factoids')
            print(all_resp)
            all_resp = [f"{i+1}. {st}" for i, st in enumerate(all_resp)]
            all_resp.insert(0, '### FACTOIDS:\n')
            #all_topics = list(dict.fromkeys(all_topics))
            #all_resp.insert(0, '### TOPICS: ' + ",".join(all_topics))
            with open(txt_file, 'w') as fp:
                fp.writelines(all_resp)
                #fp.write(f'Factoids: \n{summary}\n\nTime taken: {(time() - st)/60} mins')
            #print("\nSummary:\n", summary)
        elif instruction_type == INSTRUCTION_TYPES['QUERIES']:
            factoid_file = f'data/factoids/factoids-{self.filename}.txt'
            with open(factoid_file, 'r') as fp:
                file_contents = fp.read()
            self.__generate_query(file_contents)
            #self.__cleanup_queries()
        else:
            print('Passed invalid instruction type!')
            SystemExit()
            
        print(f'\n\n### TIME TAKEN: {(time() - st)/60} mins')
        sys.stdout = old_stdout
        log_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type = int, default = 1, required = False)
    parser.add_argument('--min_factoids', type = int, default = 5, required = False)
    parser.add_argument('--max_factoids', type = int, default = 10, required = False)
    parser.add_argument('--max_chunk_length', type = int, default = 5000, required = False)
    parser.add_argument('--max_tokens', type = int, default = 4096, required = False)
    parser.add_argument('--instruction_type', type = str, default = INSTRUCTION_TYPES['FACTOIDS'], required = False)

    args = parser.parse_args()

    filename = '000000248824000012-amd-20231230'
    bu_query_gen = BottomUpQueryGenerator(filename = filename, model_index = args.model_index,
        chunk_length = args.max_chunk_length, max_tokens = args.max_tokens,
        min_factoids = args.min_factoids, max_factoids = args.max_factoids)
    bu_query_gen.run(instruction_type=args.instruction_type)
