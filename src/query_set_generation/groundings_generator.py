import os
import torch
import multiprocessing
from vllm import LLM
import json
from time import time
import sys
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc
from google import genai

from utils.string_utils import extract_json_array_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, execute_gemini_LLM_task, execute_llama_LLM_task, get_tokenizer

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
    "gemini-2.0-flash"
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

class GroundingsGenerator:

    def __init__(self, model_index = 6, prompt_batch_size = 3):
        self.model_name = MODELS[model_index]
        self.prompt_batch_size = prompt_batch_size

        with open("./config.json", "r") as fp:
            cfg = json.load(fp)
        self.device = torch.device("cuda")

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
                api_key = cfg["google_api_keys"]["vinitramk1"]
            )
            self.model_folder = "gemini"
        elif "Llama-3.3-70B" in self.model_name:
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
        else:
            raise SystemExit('Invalid model index passed!')

    def __generate_groundings(self, fact_doc_texts, factoids_arr_batch, query_strs, metadata):

        grounding_instruction_prompt = """
        ### Task:
        Analyze the provided question, set of factoids with their citations and the metadata about them, and generate groundings for the question and answer pair.
        Groundings are citations of the factoids that support or are relevant to the question.

        ### Generation Rules
        - **Do not put gibberish, unnecessary, ellaborate adjectives and chinese characters** in your response for either question or the answer.
        - **Do not put opinions, your intermediate reasoning steps used in forming the response**.
        - The groundings should be citations picked directly from the provided citations in the input prompt.
        - **Do not** generate new factoids to put in the groundings to support the answer.
        - **Do not** put incorrect punctuations in the factoids.
        - Use the example structure as reference to return the final response. **Do not copy example from the prompt** in your response.
        - Return clean groundings with no typos, grammatical mistakes or erronuous punctuations.
        - Phrase your response as concisely as possible, in English only.
        - **Don't think** for more than 2000 tokens

        ### Input format:
        Question: <question text>
        Answer: <answer text>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids with citations\>]

        ### Output format:
        "groundings": [\<list of citations of factoids, that support the answer\>]

        ### Example Output (JSON):
        "groundings": [
            "Apple has committed to achieving carbon neutrality across its entire business, including supply chain and product life cycle, by 2030.",
            "Apple invests in renewable energy and low-carbon manufacturing processes as part of its environmental sustainability goals.",
            "The company requires its suppliers to comply with its environmental standards, including carbon reduction initiatives.",
            "Apple works with suppliers to transition to clean energy and energy-efficient production methods.",
            "The company integrates recycled and sustainable materials into product design and manufacturing.",
            "Apple acknowledges that its environmental initiatives may lead to higher costs in the short term due to increased material and compliance expenses.",
            "The company anticipates that its ESG efforts will improve brand reputation and customer loyalty.",
            "Apple views its leadership in ESG initiatives as a competitive advantage, positioning it to mitigate future regulatory and environmental risks."
        ]
        
        ### Input for your task:
        """

        zipped_query_factoids = list(zip(query_strs, fact_doc_texts))
        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nQuery: {qo[0]}\nMetadata: {metadata}\nFactoids: {qo[1]}" for qo in zipped_query_factoids]
        qna_pairs_gen = []
        missed_qna_pairs = []
        zipped_query_factoids = list(zip(query_strs, factoids_arr_batch))
        if "gemini" in self.model_name:
            for gi, grounding_instruction_prompt in enumerate(grounding_instruction_prompts):
                gsummary = execute_gemini_LLM_task(self.llm, grounding_instruction_prompt)
                print(f'generated response for question: ', gsummary)
                gjson_arr = extract_json_array_by_key(gsummary, "groundings")
                if gjson_arr != None and len(gjson_arr) > 0:
                    qna_pairs_gen.append({
                        "query": zipped_query_factoids[gi][0],
                        "factoids": zipped_query_factoids[gi][1],
                        "groundings": gjson_arr
                    })
                else:
                    missed_qna_pairs.append({
                        "query": zipped_query_factoids[gi][0],
                        "factoids": zipped_query_factoids[gi][1],
                    })
        elif "Llama-3.3-70B" in self.model_name:
            grounding_system_prompt = "You are a helpful assistant that given a question and set of factoids & citations, returns groundings (citations supporting the answer)."
            grounding_prompt_tokens = self.tokenizer([get_prompt_token(grounding_instruction_prompt, grounding_system_prompt, self.tokenizer) for grounding_instruction_prompt in grounding_instruction_prompts], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            goutputs = execute_llama_LLM_task(self.llm, grounding_prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            #print('test response llama', goutputs[0])
            for zqf, o in zip(zipped_query_factoids, goutputs):
                gsummary = o
                print(f'generated response for question: ', gsummary)
                if "Input for your task" in gsummary:
                    ti = gsummary.index("Input for your task")
                    gjson_arr = extract_json_array_by_key(gsummary[ti:], "groundings")
                    print('qjson', gjson_arr)
                    if gjson_arr != None and len(gjson_arr) > 0:
                        qna_pairs_gen.append({
                            "query": zqf[0],
                            "factoids": zqf[1],
                            "groundings": gjson_arr
                        })
                    else:
                        missed_qna_pairs.append({
                            "query": zqf[0],
                            "factoids": zqf[1]
                        })
        else:
            grounding_system_prompt = "You are a helpful assistant that given a question and set of factoids & citations, returns groundings (citations supporting the answer)."
            grounding_prompt_tokens = [get_prompt_token(gip, grounding_system_prompt, self.tokenizer) for gip in grounding_instruction_prompts]
            goutputs = execute_LLM_tasks(self.llm, grounding_prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)

            for zqf, o in zip(zipped_query_factoids, goutputs):
                gsummary = o.outputs[0].text.strip()
                print(f'generated response for question: ', gsummary)
                gjson_arr = extract_json_array_by_key(gsummary, "groundings")
                if gjson_arr != None and len(gjson_arr) > 0:
                    qna_pairs_gen.append({
                        "query": zqf[0],
                        "factoids": zqf[1],
                        "groundings": gjson_arr
                    })
                else:
                    missed_qna_pairs.append({
                        "query": zqf[0],
                        "factoids": zqf[1],
                    })

        print('no of valid qna and grounding pairs', len(qna_pairs_gen))
        print('no of invalid qna and grounding pairs', len(missed_qna_pairs))

        return qna_pairs_gen, missed_qna_pairs
    
    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        
    def generate_groundings(self, topic_index = 0):

        all_resp = []

        iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
        
        #chunk_obj = [chunk for chunk in chunk_store["chunks"] if chunk["chunk_filename"] == chunk_fn]
        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of queries formed: ', len(query_arr))
            filtered_queries = [querysets for querysets in query_arr if querysets["topic"] == SEED_METADATA_TOPICS[topic_index]]
            if len(filtered_queries) > 0:
                filtered_queries = filtered_queries[0]["query_sets"]
            else:
                print(f'no queries formed for the topic: {SEED_METADATA_TOPICS[topic_index]}')
                SystemExit()

            print('total length of filtered array: ', len(filtered_queries))
            metadata = f'Company: {self.company_name} | SEC Filing: 10-K | Related topic: {SEED_METADATA_TOPICS[topic_index]}'
            all_resp = []
            print('\nStarting grounding generation for batch of factoids\n')
            for bi,i in enumerate(range(0, len(filtered_queries), self.prompt_batch_size)):
                query_strs = [qs['query'] for qs in filtered_queries[i:(i+self.prompt_batch_size)]]
                factoids_arr_batch = [qs["factoids"] for qs in filtered_queries[i:(i+self.prompt_batch_size)]]
                factoids_arr_str_batch = [json.dumps(qs, indent=4) for qs in factoids_arr_batch]
                #factoids_doc_batch = [factoid_subarr_str + "]" for factoid_subarr_str in factoids_arr_str_batch]
                print(f'\nRunning grounding generation for factoids batch {bi}')
                qobjs, missed_qna_pairs = self.__generate_groundings(factoids_arr_str_batch, factoids_arr_batch, query_strs, metadata)
                all_resp.extend(qobjs)
                attempts = 0
                while (len(missed_qna_pairs) != 0) and (attempts < NO_OF_TRIALS):
                    query_strs = [qs['query'] for qs in filtered_queries[i:(i+self.prompt_batch_size)]]
                    factoids_arr_batch = [qs["factoids"] for qs in filtered_queries[i:(i+self.prompt_batch_size)]]
                    factoids_arr_str_batch = [json.dumps(qs, indent=4) for qs in factoids_arr_batch]
                    print(f'\nRunning grounding generation for factoids batch {bi}')
                    qobjs, missed_qna_pairs = self.__generate_groundings(factoids_arr_str_batch, factoids_arr_batch, query_strs, metadata)
                    all_resp.extend(qobjs)
                    attempts += 1
                    
            print('No of valid query, answer and grounding set: ', len(all_resp))
            if len(all_resp) > 0:
                topic_queries = [tq for tq in query_store["queries"] if tq["topic"] == SEED_METADATA_TOPICS[topic_index]]
                if len(topic_queries) > 0:
                    topic_queries = topic_queries[0]
                    topic_queries["query_sets"] = all_resp
                    for iq,_ in enumerate(query_store["queries"]):
                        if query_store["queries"][iq]["topic"] == SEED_METADATA_TOPICS[topic_index]:
                            query_store["queries"][iq]["query_sets"] = topic_queries["query_sets"]
                else:
                    topic_queries = { "topic": SEED_METADATA_TOPICS[topic_index], "query_sets": [] }
                    topic_queries["query_sets"] = all_resp
                    query_store["queries"].append(topic_queries)

                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp) 
        else:
            raise SystemExit('Chunk store not found!')

    def destroy(self):
        #del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        #os._exit(0)

if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-groundings-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 3, required = False)

    args = parser.parse_args()

    ground_gen = GroundingsGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating groundings for file: {args.filename}')
    ground_gen.set_filename(args.filename)
    if args.topic_index == -1:
        for ti in range(len(SEED_METADATA_TOPICS)):
            print(f'\nGenerating groundings for qna pairs on topic {SEED_METADATA_TOPICS[ti]}')
            ground_gen.generate_groundings(topic_index = ti)
            print(f'Finished generating groundings for topic {SEED_METADATA_TOPICS[ti]}')
            torch.cuda.empty_cache()
    else:
        print(f'\nGenerating groundings for qna pairs on topic {SEED_METADATA_TOPICS[args.topic_index]}')
        ground_gen.generate_groundings(topic_index = args.topic_index)
        print(f'Finished generating groundings for topic {SEED_METADATA_TOPICS[args.topic_index]}')

    ground_gen.destroy()
    
    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    os._exit(0)