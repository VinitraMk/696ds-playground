import os
import torch
from vllm import LLM
import json
from time import time
import sys
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc
from google import genai
from together import Together
import random

from utils.string_utils import extract_json_array_by_key, is_valid_sentence
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

IGNORE_ENTITIES = ['Table of Contents', 'SEC', '10-K filings', 'SEC 10-K filings', 'SEC 10-K', 'SEC (Securities and Exchange Commission)', 'Notes', 'Item 1A', 'Part IV, Item 15', 'Item 601(b)(32)(ii)', 'Item 15', 'Item']

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
        
    def __extract_and_clean_groundings(self, sentences):
        #factoid_list = extract_json_object_array_by_keys(text, ["factoid", "citation"])
        if len(sentences) > 0:
        #print('factoid list', factoid_list)
            clean_sentences = [s for s in sentences if is_valid_sentence(s, 200)]
        else:
            clean_sentences = []
        return clean_sentences

    def __generate_groundings(self, chunk_texts, entity, metadata):

        grounding_instruction_prompt = """
        ### Task:
        Analyze the provided chunk of text, the entity and the metadata about the text and generate a detailed 
        summary about sentences from the chunk of text addressing the entity, directly or indirectly. This summary is called a grounding.

        ### Generation Rules
        - **Do not put gibberish, unnecessary, ellaborate adjectives and chinese characters** in your response for either question or the answer.
        - **Do not put opinions, your intermediate reasoning steps used in forming the response**.
        - The groundings can be short or long. The summary of the sentences related to the given entity (grounding) should be detailed and clear, covering
        the necessary background or context around the sentences as well.
        - Generate the groundings in as much detail as possible, but keep each grounding below 200 words.
        - Phrase your response in **English only.**
        - **Do not** generate completely new groundings that are not addressed in the given text.
        - **Do not** put incorrect punctuations in the groundings.
        - Use the example structure as reference to return the final response. **Do not copy example from the prompt** in your response.
        - Return clean groundings with no typos, grammatical mistakes or erronuous punctuations.
        - **Don't think** for more than 2000 tokens.

        ### Input format:
        Text: <chunk of text from SEC filing>
        Entity: <entity>
        Metadata: <meta data of the main company from whose SEC 10-K filing the chunk of text is from>

        ### Output format:
        "groundings": [\<list of sentences, that related to the entity\>]

        ### Example:
        "groundings": [
            "The 10-K filing notes that 'The Company’s business, results of operations and financial condition could be materially adversely affected by changes in global economic conditions.' It also states that 'The Company is subject to intense competition in all markets in which it operates,' highlighting exposure to industry dynamics. Apple points out reliance on third-party suppliers and manufacturers, stating, 'The Company depends on component and product manufacturing and logistical services provided by outsourcing partners.",
            "Net sales increased 8% or $29.3 billion during 2023 compared to 2022' indicates strong performance, particularly in the iPhone and Services segments. Apple adds, 'Research and development expense increased to $27.7 billion in 2023,' showing commitment to innovation. The filing explains margin variability with 'We expect gross margin to fluctuate in future periods, depending on a variety of factors, including product mix and component costs.",
            "As of September 30, 2023, the Company’s cash, cash equivalents and marketable securities totaled $162.1 billion' signals substantial liquidity. Apple mentions capital allocation strategies in 'The Company’s capital return program includes both share repurchases and dividends.' The filing also adds, 'The Company believes its existing cash, cash equivalents and marketable securities, together with cash generated from operations, will be sufficient to meet its liquidity needs.",
            "Apple outlines its sustainability goals with the statement 'The Company is committed to achieving carbon neutrality across its entire business by 2030.' It also includes, 'Our environmental programs focus on reducing emissions, improving material recovery, and using recycled content in our products and packaging.' These disclosures reflect Apple's broader ESG strategy and long-term environmental impact planning.",
            "The Company is subject to taxation in the U.S. and numerous foreign jurisdictions' indicates ongoing global tax compliance exposure. It further notes, 'Apple is involved in legal proceedings and investigations from time to time, including antitrust matters in multiple regions,' reflecting regulatory scrutiny. These matters may materially affect financial performance or require changes to business operations depending on their outcomes."
        ]
        
        ### Input for your task:
        """

        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nText: {chunk['text']}\nEntity: {entity}\nMetadata: {metadata}" for chunk in chunk_texts]
        groundings_set = {}
        if "gemini" in self.model_name:
            for gi, grounding_instruction_prompt in enumerate(grounding_instruction_prompts):
                groundings = []
                gsummary = execute_gemini_LLM_task(self.llm, grounding_instruction_prompt)
                print(f'generated response for question: ', gsummary)
                gjson_arr = extract_json_array_by_key(gsummary, "groundings")
                if gjson_arr != None and len(gjson_arr) > 0:
                    clean_g = self.__extract_and_clean_groundings(gjson_arr)
                    for gc in clean_g:
                        groundings.append({ 'text': gc, 'entity': entity})
                groundings_set[chunk_texts[gi]['chunk_index']] = groundings
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            grounding_system_prompt = "You are a helpful assistant that given a question and set of factoids & citations, returns groundings (citations supporting the answer)."
            grounding_prompt_tokens = self.tokenizer([get_prompt_token(grounding_instruction_prompt, grounding_system_prompt, self.tokenizer) for grounding_instruction_prompt in grounding_instruction_prompts], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            goutputs = execute_llama_LLM_task(self.llm, grounding_prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            #print('test response llama', goutputs[0])
            for gi,o in enumerate(goutputs):
                groundings = []
                gsummary = o
                print(f'generated response for question: ', gsummary)
                if "Input for your task" in gsummary:
                    ti = gsummary.index("Input for your task")
                    gjson_arr = extract_json_array_by_key(gsummary[ti:], "groundings")
                    print('qjson', gjson_arr)
                    if gjson_arr != None and len(gjson_arr) > 0:
                        clean_g = self.__extract_and_clean_groundings(gjson_arr)
                        for gc in clean_g:
                            groundings.append({ 'text': gc, 'entity': entity} )
                groundings_set[chunk_texts[gi]['chunk_index']] = groundings
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            grounding_system_prompt = "You are a helpful assistant that given a question and set of factoids & citations, returns groundings (citations supporting the answer)."
            for gi, grounding_instruction in enumerate(grounding_instruction_prompts):
                groundings = []
                gsummary = execute_llama_task_api(self.llm, grounding_instruction, grounding_system_prompt)
                print('generated response: ', gsummary)
                gjson_arr = extract_json_array_by_key(gsummary, "groundings")
                if gjson_arr != None and len(gjson_arr) > 0:
                    clean_g = self.__extract_and_clean_groundings(gjson_arr)
                    for gc in clean_g:
                        groundings.append({'text': gc, 'entity': entity})
                groundings_set[chunk_texts[gi]['chunk_index']] = groundings
        else:
            grounding_system_prompt = "You are a helpful assistant that given a question and set of factoids & citations, returns groundings (citations supporting the answer)."
            grounding_prompt_tokens = [get_prompt_token(gip, grounding_system_prompt, self.tokenizer) for gip in grounding_instruction_prompts]
            goutputs = execute_LLM_tasks(self.llm, grounding_prompt_tokens, max_new_tokens=8192, temperature=0.6, top_p=0.9)

            for gi, o in enumerate(goutputs):
                gsummary = o.outputs[0].text.strip()
                print(f'generated response for question: ', gsummary)
                gjson_arr = extract_json_array_by_key(gsummary, "groundings")
                if gjson_arr != None and len(gjson_arr) > 0:
                    clean_g = self.__extract_and_clean_groundings(gjson_arr)
                    for gc in clean_g:
                        groundings.append({
                            'text': gc,
                            'entity': entity
                        })
                groundings_set[chunk_texts[gi]['chunk_index']] = groundings

        return groundings_set
    
    
    def __sample_entities(self, entities_info, count_range = (5, 15), k = 10):
        relevant_entities = {ek: entities_info[ek] for ek in entities_info.keys() if (ek not in IGNORE_ENTITIES) and (entities_info[ek]['count'] >= count_range[0]) and (entities_info[ek]['count'] <= count_range[1])}
        sampled_entities = random.sample(relevant_entities.keys(), k)
        return sampled_entities

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]

    def generate_groundings(self):


        chunk_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_chunk_store.json'
        entity_store_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_entities_info.json'
        sampled_entities_fp = f'data/chunked_data/global_chunk_store/{self.model_folder}/{self.filename}_sampled_entities.json'
        chunks_obj_fp = f'data/chunked_data/{self.filename}_chunked.json'

        if os.path.exists(chunk_store_fp) and os.path.exists(entity_store_fp):
            with open(chunk_store_fp, 'r') as fp:
                chunk_store = json.load(fp)
            chunk_store_arr = chunk_store["chunks"]

            with open(entity_store_fp, 'r') as fp:
                entity_store = json.load(fp)

            with open(chunks_obj_fp, 'r') as fp:
                chunks_obj = json.load(fp)
            chunk_arr = chunks_obj['chunks']

            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting grounding generation for each chunk\n')

            sampled_entity_keys = self.__sample_entities(entities_info=entity_store, count_range=(5, 25), k = 20)
            print('\nSampled entities: ', sampled_entity_keys)
            with open(sampled_entities_fp, 'w') as fp:
                json.dump({ "sampled_entities": sampled_entity_keys }, fp)

            for ek in sampled_entity_keys:
                chunk_indices = entity_store[ek]["chunk_indices"]
                #chunk_entities = chunk_store_arr[ci]['entities']
                for cix in range(0, len(chunk_indices), self.prompt_batch_size):
                    cix_batch = chunk_indices[cix:cix+self.prompt_batch_size]
                    chunk_texts = [{ 'chunk_index': ci, 'text': chunk_arr[ci]} for ci in cix_batch]
                    entity_groundings = self.__generate_groundings(chunk_texts = chunk_texts, entity=ek, metadata=metadata)
                    for ci in cix_batch:
                        if 'groundings' not in chunk_store_arr[ci]:
                            chunk_store_arr[ci]['groundings'] = entity_groundings[ci]
                        else:
                            chunk_store_arr[ci]['groundings'].extend(entity_groundings[ci])

            chunk_store["chunks"] = chunk_store_arr
            with open(chunk_store_fp, 'w') as fp:
                json.dump(chunk_store, fp) 
        else:
            raise SystemExit('Chunk store not found!')

    def destroy(self):
        #del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        os._exit(0)

if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-groundings-logs.txt'
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

    ground_gen = GroundingsGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating groundings for file: {args.filename}')
    ground_gen.set_filename(args.filename)
    ground_gen.generate_groundings()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    ground_gen.destroy()