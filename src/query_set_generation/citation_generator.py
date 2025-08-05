import os
import torch
from vllm import LLM
import multiprocessing
import json
from time import time, sleep
import sys
import argparse
import re
import gc
from google import genai
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from together import Together
from groq import AsyncGroq

from utils.string_utils import is_valid_sentence, extract_json_text_by_key, extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer, execute_llama_task_api, execute_groq_task_api
from src.prompts.query_set_generation.citation_prompt import CITATION_INSTRUCTION_PROMPT
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS, HF_CACHE_DIR


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
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            self.llm = AsyncGroq(api_key = cfg["groq_api_key"])
            self.model_folder = "llama"
        else:
            raise SystemExit('Invalid model index passed!')

    def __get_output_from_llm(self, instruction_prompts, system_prompt, json_schema = None, llm_config = None):
        summary = ""
        if self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            prompt_tokens = self.tokenizer([get_prompt_token(instruction_prompts[0], system_prompt, self.tokenizer)], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            outputs = execute_llama_LLM_task(self.llm, prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            summary = outputs[0]
            if "Input for your task" in summary:
                ti = summary.index("Input for your task")
                summary = summary[ti:]
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            summary = execute_llama_task_api(self.llm, instruction_prompts[0], system_prompt)
            print('generated response: ', summary)
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            summary = execute_groq_task_api(self.llm, json_schema, instruction_prompts, system_prompt)
            summary = [robj['response'] for robj in summary]
            print('generated response: ', summary[0])
        else:
            prompt_tokens = [get_prompt_token(instruction_prompts[0], system_prompt, self.tokenizer)]
            outputs = execute_LLM_tasks(self.llm, prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)
            summary = outputs[0].outputs[0].text.strip()
            print(f'generated response: ', summary)

        return summary

    def __generate_citations(self, chunk, qna_pair, metadata):

        citation_system_prompt = "You are a helpful assistant, that given a chunk of text, a Q&A pair and metadata about the company addressed in the Q&A pair, extracts citations from the chunk of text that support the answer to the question."
        citation_instruction_prompt = CITATION_INSTRUCTION_PROMPT
        citation_instruction_prompt = citation_instruction_prompt + f"\nText:{chunk}\nQuery: {qna_pair['query']}\nAnswer: {qna_pair['answer']}\nMetadata: {metadata}"
        citations = []
        if self.model_name == "meta-llama/llama-3.3-70b-versatile":
            citation_json_schema = {
                "type": "json_object",
                "name": "citation_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["citations"],
                    "additionalProperties": False
                }
            }
        else:
            citation_json_schema = None
        csummary = self.__get_output_from_llm([citation_instruction_prompt], citation_system_prompt, citation_json_schema)
        cjson = extract_json_array_by_key(csummary[0], "citations")
        if cjson != None and len(cjson) > 0:
            citations = cjson
               
        return citations

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        
    def generate_citations(self, no_of_entities = 20):
        
        iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_generated_queries.json'
        chunk_store_fp = f'data/chunked_data/chunks/{self.filename}_chunked.json'
        main_query_store_fp = f'data/queries/{self.model_folder}/{self.filename}_generated_queries.json'
        subpar_query_store_fp = f'data/queries/{self.model_folder}/{self.filename}_subpar_queries.json'

        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of queries formed: ', len(query_arr))

            with open (chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)["chunks"]

            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting answer generation for batch of questions\n')
            #sampled_entities = list(query_arr.keys())
            sampled_entities = ["Ai", "Intangible Assets", "Data Center"]
            print('\nSampled entities: ', sampled_entities)
            less_hop_qstns = []
            for ei in range(no_of_entities):
                entity = sampled_entities[ei]
                filtered_queries = query_arr[entity]
                for qi, query_obj in enumerate(filtered_queries):
                    chunks_used = query_obj["chunks_used"]
                    docs_considered = query_obj['docs_considered']

                    all_citations = []
                    cited_chunks = []
                    qna_pair = { 'query': query_obj['query'], 'answer': query_obj['answer'] }
                    for cmp in docs_considered:
                        rel_chunks = [cobj['chunk_index'] for cobj in chunks_used if cobj['company_code'] == cmp]
                        cmp_filename = COMPANY_DICT[cmp]['filename']
                        with open(os.path.join('data/chunked_data/chunks', f'{cmp_filename}_chunked.json'), 'r') as fp:
                            cmp_chunk_store = json.load(fp)["chunks"]
                        for ci in rel_chunks:
                            chunk = cmp_chunk_store[ci]
                            chunk_citations = self.__generate_citations(chunk = chunk, qna_pair = qna_pair, metadata = metadata)
                            print('generated citations', len(chunk_citations), chunk_citations)
                            if len(chunk_citations) > 0:
                                all_citations.extend(chunk_citations)
                                cited_chunks.append({ 'company_name': cmp, 'chunk_index': ci})
                    if len(cited_chunks) < 5:
                        less_hop_qstns.append({ 'entity': entity, 'query': query_obj['query'], 'answer': query_obj['answer'], 'groundings': query_obj['groundings'], 'citations': all_citations, 'chunks_used': cited_chunks })
                    filtered_queries[qi] = query_obj | { 'citations': all_citations, 'chunks_used': cited_chunks }
                filtered_queries = [query_obj for query_obj in filtered_queries if "citations" in query_obj and len(query_obj["citations"]) > 0]

                #sleep(90)
                query_arr[entity] = filtered_queries                    
                query_store["queries"] = query_arr
                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp) 

            with open(subpar_query_store_fp, 'w') as fp:
                json.dump({ "queries": less_hop_qstns}, fp)

            print('\nNo of questions with less than 5 hops: ', len(less_hop_qstns))

            
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
    parser.add_argument('--no_of_entities', type = int, default = 20, required = False)

    args = parser.parse_args()

    ans_gen = CitationGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating answers for file: {args.filename}')
    ans_gen.set_filename(args.filename)
    ans_gen.generate_citations(no_of_entities = args.no_of_entities)
    torch.cuda.empty_cache()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

    ans_gen.destroy()
