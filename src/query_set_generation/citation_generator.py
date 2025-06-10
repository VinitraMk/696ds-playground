import os
import torch
from vllm import LLM
import multiprocessing
import json
from time import time
import sys
import argparse
import re
import gc
from google import genai
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from together import Together

from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer, execute_llama_task_api


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
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
    "gemini-2.0-flash",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
]

SEED_METADATA_TOPICS = [
    "Risk Factors and Challenges",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
    "Market Trends, Economic Environment, and Industry Dynamics"
]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR

RELEVANCE_THRESHOLD = 2.0
MAX_FACTOIDS_TO_SAMPLE = 25
MIN_FACTOIDS_NEEDED_FOR_GENERATION = 15
PROMPT_BATCH_SIZE = 1
NO_OF_TRIALS = 3
FILENAMES = [
    '10-K_AMD_20231230',
    '10-K_NVDA_20240128',
    '10-K_F_20231231',
    '10-K_GM_20231231',
    '10-K_INTC_20231230',
    '10-K_TSLA_20231231'
]

class CitationGenerator:

    def __init__(self, model_index = 6, prompt_batch_size = 3):
        self.model_name = MODELS[model_index]
        self.prompt_batch_size = prompt_batch_size

        self.device = torch.device("cuda")
        with open("./config.json", "r") as fp:
            cfg = json.load(fp)

        if "QwQ" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                    quantization = "awq",
                    download_dir = HF_CACHE_DIR,
                    max_model_len = 2048 * 4,
                    #gpu_memory_utilization=0.95,
                    tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        elif "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "qwq"
        elif "gemini" in self.model_name:
            self.llm = genai.Client(
                api_key = cfg["google_api_keys"]["vinitramk4"]
            )
            self.model_folder = "gemini"
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            model_path = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True,  
                llm_int8_enable_fp32_cpu_offload=True
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,  
                device_map="sequential",  
                offload_folder="/tmp/offload", 
                local_files_only=True
            )
            self.model_folder = "llama"
            self.tokenizer = get_tokenizer(self.model_name)
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            self.llm = Together(api_key = cfg["togetherai_api_key"])
            self.model_folder = "llama"
        else:
            raise SystemExit('Invalid model index passed!')

    def __generate_citations(self, chunk, qna_pair, metadata):

        citation_instruction_prompt = """
        ### Task:
        Analyze the provided chunk of text, question-answer pair and the metadata about company about which the Q&A pair is,
        and return the exact sentence or sentences that support the answer to the question.

        ### Answer Generation Rules
        - **Do not put opinions, adjectives, elaborations, gibberish or unnecessary adjectives** in your response.
        - The sentences that support the answer, **must** be present in the given chunk of text.
        - The sentences from the chunk of text **must be complete sentences**.
        - **Do not put non-English characters** in your response. Return responses only in English.
        - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.

        ### Input format:
        Text: <chunk of text from SEC filing document of the company>
        Query: <query text>
        Answer: <answer text>
        Metadata: <meta data of the main company upon which the factoids are based.>

        ### Output format (JSON):
        "citations": [<list of sentences from the text supporting the answer>]

        ### Input for your task:
        """

        citation_instruction_prompt = citation_instruction_prompt + f"\nText:{chunk}\nQuery: {qna_pair['query']}\nAnswer: {qna_pair['answer']}\nMetadata: {metadata}"
        citations = []
        if "gemini" in self.model_name:
            csummary = execute_gemini_LLM_task(self.llm, citation_instruction_prompt)
            print(f'generated response for citations: ', csummary)
            cjson = extract_json_array_by_key(csummary, "citations")
            if cjson != None and len(cjson) > 0:
                citations = cjson
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            citation_system_prompt = "You are a helpful assistant, that given a chunk of text, a Q&A pair and metadata about the company addressed in the Q&A pair, extracts citations from the chunk of text that support the answer to the question."
            citation_prompt_tokens = self.tokenizer([get_prompt_token(citation_instruction_prompt, citation_system_prompt, self.tokenizer)], return_tensors = "pt", truncation = True, padding = True).to(self.device)
            coutputs = execute_llama_LLM_task(self.llm, citation_prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            csummary = coutputs[0]
            print(f'generated response for citations: ', csummary)
            if "Input for your task" in csummary:
                ti = csummary.index("Input for your task")
                cjson = extract_json_array_by_key(csummary[ti:], "citations")
                print('cjson', cjson)
                if cjson != None and len(cjson) > 0:
                    citations = cjson
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            citation_system_prompt = "You are a helpful assistant, that given a chunk of text, a Q&A pair and metadata about the company addressed in the Q&A pair, extracts citations from the chunk of text that support the answer to the question."
            csummary = execute_llama_task_api(self.llm, citation_instruction_prompt, citation_system_prompt)
            print('generated response: ', csummary)
            cjson = extract_json_array_by_key(csummary, "citations")
            if cjson != None and len(cjson) > 0:
                citations = cjson
        else:
            citation_system_prompt = "You are a helpful assistant, that given a chunk of text, a Q&A pair and metadata about the company addressed in the Q&A pair, extracts citations from the chunk of text that support the answer to the question."
            citation_prompt_tokens = [get_prompt_token(citation_instruction_prompt, citation_system_prompt, self.tokenizer)]
            coutputs = execute_LLM_tasks(self.llm, citation_prompt_tokens, max_new_tokens=3000, temperature=0.6, top_p=0.9)
            csummary = coutputs[0].outputs[0].text.strip()
            print(f'generated response for citations: ', csummary)
            cjson = extract_json_array_by_key(csummary, "citations")
            if cjson != None and len(cjson) > 0:
                citations = cjson
                
        return citations

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        
    def generate_citations(self):
        
        iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_generated_queries.json'
        chunk_store_fp = f'data/chunked_data/{self.filename}_chunked.json'
        main_query_store_fp = f'data/queries/{self.model_folder}/{self.filename}_generated_queries.json'

        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of queries formed: ', len(query_arr))

            if os.path.exists(main_query_store_fp):
                with open(main_query_store_fp, 'r') as fp:
                    main_query_store = json.load(fp)

            with open (chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)["chunks"]

            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting answer generation for batch of questions\n')
            sampled_entities = query_arr.keys()
            for entity in sampled_entities:
                filtered_queries = query_arr[entity]
                for qi, query_obj in enumerate(filtered_queries):
                    chunks_used = query_obj["chunks_used"]
                    all_citations = []
                    cited_chunks = []
                    qna_pair = { 'query': query_obj['query'], 'answer': query_obj['answer'] }
                    for ci in chunks_used:
                        chunk = chunk_store[ci]
                        chunk_citations = self.__generate_citations(chunk = chunk, qna_pair = qna_pair, metadata = metadata)
                        if len(chunk_citations) > 0:
                            all_citations.extend(chunk_citations)
                            cited_chunks.append(ci)
                    filtered_queries[qi] = query_obj | { 'citations': all_citations, 'chunks_used': cited_chunks }
                if entity in main_query_store["queries"]:
                    main_query_store["queries"][entity].extend(filtered_queries)
                else:
                    main_query_store["queries"][entity] = filtered_queries
                #query_arr[entity] = filtered_queries                    

                #query_store["queries"] = query_arr

                with open(main_query_store_fp, 'w') as fp:
                    json.dump(main_query_store, fp)
                os.remove(iquery_store_fp) # remove from intermediate storage after final dataset is constructed
        else:
            SystemExit('Chunk store not found!')

    def destroy(self):
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        os._exit(0)

if __name__ == "__main__":
    st = time()
    log_fp = f'logs/bu-citation-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)

    args = parser.parse_args()

    ans_gen = CitationGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating answers for file: {args.filename}')
    ans_gen.set_filename(args.filename)
    ans_gen.generate_citations()
    torch.cuda.empty_cache()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

    ans_gen.destroy()
