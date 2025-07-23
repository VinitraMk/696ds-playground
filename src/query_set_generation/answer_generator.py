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

class AnswerGenerator:

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

    def __cleanup_text(self, texts, wc = 100):
        cleanup_instruction = """
        ### Task:
        Given a text, riddled with grammatical, punctuation, insensible/gibberish words or typos (missing spaces), remove
        the errors and return a clean, grammatically correct and meaningful text if it exists. If a text doesn't exist,
        return a blank string like this - "". If there are no errors in the text, return the original input string as is.

        ### Cleanup Rules:
        - Remove punctuation mistakes such as missing spaces, incorrectly placed hyphens, obvious typos, etc.
        - Return a grammatically correct and meaningful sentence. If it doesn't exist, return a blank string like this - "".
        - Remove chinese characters.
        - If the sentence contains gibberish, insensible words return a blank string like this - "".
        - **Do not** hallucinate completely new words in the sentence.
        - **Do not** put chinese words in the sentence.
        - If the text is already meaningful and sensible, return the original input string as is.

        ### Input format:
        Text: <text>

        ### Output format:
        Cleaned Text: <cleaned up sentence>

        ### Example Input:
        Text: "USSGovernmentIntroducedLicensingRequirementsImpactingeExportsToChinHongKonMaouanRussiaIncludingAIICAndHIIntegratedCircuDuringThThirdQuarterOfFiscYearII"

        ### Example Output:
        Cleaned Text: "US Government introduced licensing requirements impacting exports to China, Hong Kong, Macau, and Russia, including AI IC and HI integrated circuits during the third quarter of Fiscal Year II."

        ### Example Input:
        Text: "To create an intricate inquiry leveraging several provided facts about Nvidia (NVDA), I focused primarily around how regulatory constraints tied directly back toward broader corporate vulnerabilities stemming partly out environment-centric policies/risks; especially those involving governmental oversight concerning international trade alongside internal pressures arising because excessive resource usage patterns may provoke additional scrutiny/litigation threats down road if mishandles appropriately enough moving forward strategically speaking wise-wise mannerism approachable way feasible indeed possible certainly achievable realistically attainably feasibly plausible logically reasonable sensically rationally understandably comprehensibly coherently consistently maintainingly sustainabily durablly persistantly continuously perpetually enduringlty lastling"

        ### Example Output:
        Cleaned Text: ""

        ### Example Input:
        Text: "Apple works with suppliers to transition to clean energy and energy-efficient production methods."

        ### Example Output:
        Cleaned Text: "Apple works with suppliers to transition to clean energy and energy-efficient production methods."

        ### Input for your task:
        """

        cleaned_sentences = []
        cleanup_instructions = [cleanup_instruction + f"\nText: \"{text}\"" for text in texts]
        if "gemini" in self.model_name:
            osummary = execute_gemini_LLM_task(self.llm, cleanup_instructions[0])
            match = re.search(r'Cleaned\s+Text:\s*"([^"]+)"', osummary)
            if (match and is_valid_sentence(match.group(1), wc)):
                cleaned_sentences.append(match.group(1))
            else:
                cleaned_sentences.append("")
            return cleaned_sentences
        else:
            cleanup_system_prompt = "You are a helpful assistant, that given a text returns a cleaned, grammatically meaningful sentence if it exists."
            cleanup_instruction_prompt_tokens = [get_prompt_token(prompt, cleanup_system_prompt, self.tokenizer) for prompt in cleanup_instructions]
            coutputs = execute_LLM_tasks(self.llm, prompts=cleanup_instruction_prompt_tokens, max_new_tokens=2000, temperature=0.1, top_p=0.9)
            for o in coutputs:
                osummary = o.outputs[0].text.strip()
                print('generated clean up response', osummary)
                match = re.search(r'Cleaned\s+Text:\s*"([^"]+)"', osummary)
                if (match and is_valid_sentence(match.group(1), wc)):
                    cleaned_sentences.append(match.group(1))
                else:
                    cleaned_sentences.append("")
            return cleaned_sentences

    def __refine_answers(self, factoids_doc_text, factoids_arr_batch, qna_pairs, metadata):
        refineans_instruction_prompt = """
        ### Task:
        Given a query-answer pair, some metadata, a list factoids, refine the answer
        to the question using the factoids in the prompt. Also return the list of factoids used to reframe the answer.

        ### Answer Generation Rules
        - **No opinions, adjectives, elaborations or extra details**
        - Keep the final refined answer precise, summarizing the factoids.
        - Newly refined answer is ONLY from the given factoids, **do not** hallucinate new information or factoids to answer the query.
        - Return the final answer as one single concise paragraph of under 200 words in a json object. Use the example output as reference for structure.
        - **Do not put chinese characters** in your response. Return responses only in English.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response.
        - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
        - **Do not copy example from the prompt** in your response.
        - Don't think for more than 3000 tokens.

        ### Input format:
        Query: <query text>
        Answer: <answer text>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of citations relevant to the Q&A pair.\>]

        ### Output format (JSON):
        Answer: {
            "answer": <answer to the question, generated from fact(s) in the given text document>
        }

        ### Example Input
        Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?"
        Answer: "Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs. On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation. From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments."
        Metadata: Company name: Apple | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Factoids: [
            "We set an ambitious goal — to make our products carbon neutral by 2030, across our entire supply chain and the lifetime energy use of our customers’ devices.",
            "Our corporate operations have run on 100% renewable energy since 2018.",
            "Apple also praised its continuing work in recycling, and making new components out of recycled materials. In 2023, 56% of cobalt in Apple batteries came from recycled sources, a 2x increase compared to the previous year.",
            "Apple is calling on its suppliers to decarbonize operations as the tech giant looks to become carbon neutral by 2030. The company is asking manufacturers to decarbonize Apple-related operations by taking steps such as running on 100% renewable electricity.",
            "Apple plans to invest in renewable energy projects for its suppliers, emissions reduction technologies, product redesigns and recycling techniques, sustainable sourcing practices, and carbon removal projects."
        ]

        ### Example Output:
        Answer: {
            "answer": "Apple’s commitment to achieving carbon neutrality across its entire supply chain and products by 2030, as disclosed in its 10-K, has far-reaching implications for its cost structure, supplier relationships, and long-term profitability. The company has made significant sustainability-driven investments, such as sourcing renewable energy for global operations and integrating recycled materials into product design. These efforts have led to increased upfront costs, reflecting capital expenditures on clean energy infrastructure, low-carbon manufacturing, and material innovation. On the supply chain front, Apple works closely with its partners to enforce carbon reduction targets, which introduces higher compliance costs. This can be particularly challenging for smaller or less-resourced suppliers, potentially creating supply chain risks if partners fail to meet Apple’s environmental standards. However, by fostering collaborations with environmentally aligned suppliers, Apple enhances long-term supplier resilience, innovation potential, and operational synergy. From a profitability perspective, while short-term margins may be compressed due to the elevated costs of sustainability initiatives, the long-term financial outlook remains favorable. These initiatives strengthen Apple’s brand equity, appeal to eco-conscious consumers, and support premium pricing. Additionally, early adoption of robust ESG practices positions Apple to mitigate future regulatory risks, such as carbon taxation or emission-based trade restrictions. Thus, despite near-term financial pressures, Apple’s ESG strategy is likely to yield durable competitive and economic advantages over time.",
        }

        ### Input for your task:
        """

        #zipped_qna_groundings = list(zip(qna_pairs, factoids_doc_texts))
        rans_instruction_prompts = [refineans_instruction_prompt + f"\nQuery: {qo['query']}\nAnswer: {qo['answer']}\nMetadata: {metadata}\nFactoids: {factoids_doc_text}" for qo in qna_pairs]
        rqna_pairs = []
        #zipped_qsnts_factoids = list(zip(qna_pairs, factoids_arr_batch))
        if "gemini" in self.model_name:
            for ri, rans_instruction_prompt in enumerate(rans_instruction_prompts):
                rasummary = execute_gemini_LLM_task(self.llm, rans_instruction_prompt)
                print(f'generated response for refined answer: ', rasummary)
                ajson = extract_json_text_by_key(rasummary, "answer")
                if ajson != None and "answer" in ajson:
                    if cleaned_answers[0] != "":
                        rqna_pairs.append({
                            "query": qna_pairs[ri]['query'],
                            "answer": ajson['answer'],
                            "groundings": qna_pairs[ri]['groundings']
                        })
                    else:
                        rqna_pairs.append({
                            'query': qna_pairs[ri]['query'],
                            'answer': qna_pairs[ri]['answer'],
                            'groundings': qna_pairs[ri]['groundings']
                        })
                else:
                    rqna_pairs.append({
                        'query': qna_pairs[ri]['query'],
                        'answer': qna_pairs[ri]['answer'],
                        'groundings': qna_pairs[ri]['groundings']
                    })
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            rans_system_prompt = "You are a helpful assistant, that given a Q&A pair, a list of groundings (citations related to the given Q&A pair) improves the answer to the question based on the factoids."
            for ri, rans_instruction_prompt in enumerate(rans_instruction_prompts):
                rasummary = execute_llama_task_api(self.llm, rans_instruction_prompt, rans_system_prompt)
                print('generated response: ', rasummary)
                rajson = extract_json_text_by_key(rasummary, "answer")
                if rajson != None and "answer" in rajson:
                    rqna_pairs.append({
                        "query": qna_pairs[ri]['query'],
                        "answer": rajson['answer'],
                        "groundings": qna_pairs[ri]['groundings']
                    })
                else:
                    rqna_pairs.append({
                        "query": qna_pairs[ri]['query'],
                        "answer": qna_pairs[ri]['answer'],
                        "groundings": qna_pairs[ri]['groundings']
                    })
        else:
            rans_system_prompt = "You are a helpful assistant, that given a Q&A pair, a list of groundings (citations related to the given Q&A pair) improves the answer to the question based on the factoids."
            rans_prompt_tokens = [get_prompt_token(rans_prompt_text, rans_system_prompt, self.tokenizer) for rans_prompt_text in rans_instruction_prompts]
            raoutputs = execute_LLM_tasks(self.llm, rans_prompt_tokens, max_new_tokens=3000, temperature=0.6, top_p=0.9)

            for zqf, o in zip(qna_pairs, raoutputs):
                rasummary = o.outputs[0].text.strip()
                print(f'generated response for refined answer: ', rasummary)
                ajson = extract_json_text_by_key(rasummary, "answer")
                if ajson != None and "answer" in ajson:
                    cleaned_answers = self.__cleanup_text([ajson["answer"]], 200)
                    if cleaned_answers[0] != "":
                        rqna_pairs.append({
                            "query": zqf[0]['query'],
                            "answer": cleaned_answers[0],
                            "groundings": zqf[0]['groundings']
                        })
                    else:
                        rqna_pairs.append({
                            'query': zqf[0]['query'],
                            'answer': zqf[0]['answer'],
                            'groundings': zqf[0]['groundings']
                        })
                else:
                    rqna_pairs.append({
                        'query': zqf[0]['query'],
                        'answer': zqf[0]['answer'],
                        'groundings': zqf[0]['groundings']
                    })
            rqna_pairs = self.__refine_groundings(factoids_doc_text, factoids_arr_batch, rqna_pairs, metadata)
        return rqna_pairs

    def __generate_answer(self, qg_pair, metadata, entity):

        ans_instruction_prompt = """
        ### Task:
        Analyze the provided question, entity, list of groundings relevant to the question and the entity, and the metadata about the groudings and generate an answer to the question.

        ### Answer Generation Rules
        - **No opinions, adjectives, elaborations or extra details**
        - Generated answer is ONLY from the given groundings, **do not** hallucinate new information or groundings to answer the query.
        - Return the final answer in as much detail as possible, in a json object. Use the example output as reference for structure.
        - **Do not put non-English characters** in your response. Return responses only in English.
        - Keep the generated answer detailed and clear but keep it under 400 words.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response.
        - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
        - **Do not copy example from the prompt** in your response.
        - Don't think for more than 3000 tokens.

        ### Input format:
        Query: <query text>
        Entity: <entity>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Groundings: [\<list of citations relevant to the question and the entity\>]

        ### Output format (JSON):
        Answer: {
            "answer": <answer to the question, generated from groundings in the given text document>
        }

        ### Example Input
        Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?"
        Entity: "Apple Inc."
        Metadata: Company name: Apple | SEC Filing: 10-K
        Groundings: [
            "We set an ambitious goal — to make our products carbon neutral by 2030, across our entire supply chain and the lifetime energy use of our customers’ devices.",
            "Our corporate operations have run on 100% renewable energy since 2018.",
            "Apple also praised its continuing work in recycling, and making new components out of recycled materials. In 2023, 56% of cobalt in Apple batteries came from recycled sources, a 2x increase compared to the previous year.",
            "Apple is calling on its suppliers to decarbonize operations as the tech giant looks to become carbon neutral by 2030. The company is asking manufacturers to decarbonize Apple-related operations by taking steps such as running on 100% renewable electricity.",
            "Apple plans to invest in renewable energy projects for its suppliers, emissions reduction technologies, product redesigns and recycling techniques, sustainable sourcing practices, and carbon removal projects."
        ]

        ### Example Output:
        {
            "answer": "Apple Inc.’s commitment to achieving carbon neutrality across its supply chain and product lifecycle by 2030, as described in its 10-K filing, represents a transformative ESG strategy with implications for cost structure, supplier dynamics, and long-term profitability. Apple has set a clear target: “to make our products carbon neutral by 2030, across our entire supply chain and the lifetime energy use of our customers’ devices.” This effort spans not just internal operations—“Our corporate operations have run on 100% renewable energy since 2018”—but also imposes new standards on suppliers. The company is “asking manufacturers to decarbonize Apple-related operations by taking steps such as running on 100% renewable electricity.” This indicates a strategic push to align upstream partners with its sustainability goals. Such measures likely introduce short- to mid-term cost increases due to investments in “renewable energy projects for its suppliers, emissions reduction technologies, product redesigns and recycling techniques, sustainable sourcing practices, and carbon removal projects.” These outlays may elevate Apple’s cost structure temporarily but are framed as necessary for long-term operational resilience and brand differentiation. On the supplier side, Apple’s push for decarbonization could strain relationships with vendors unable or unwilling to transition. However, Apple mitigates this by directly supporting their transition, which may foster tighter, more collaborative partnerships over time. Apple’s recycling efforts, including the use of recycled materials—“In 2023, 56% of cobalt in Apple batteries came from recycled sources”—also reflect cost containment and risk reduction in critical material sourcing, potentially insulating Apple from raw material price volatility. While risks include higher upfront costs and execution challenges across a global supplier base, potential rewards include regulatory compliance, strengthened ESG positioning, long-term cost savings, and enhanced customer loyalty, all of which may support sustained profitability."
        }

        ### Input for your task:
        """

        groundings_str = f"[{','.join(qg_pair['groundings'])}]" 
        ans_instruction_prompt = ans_instruction_prompt + f"\nQuery: {qg_pair['query']}\nEntity: {entity}\nMetadata: {metadata}\nGroundings : {groundings_str}"
        qna_pair = {}
        if "gemini" in self.model_name:
            asummary = execute_gemini_LLM_task(self.llm, ans_instruction_prompt)
            print(f'generated response for answer: ', asummary)
            ajson = extract_json_text_by_key(asummary, "answer")
            if ajson != None and "answer" in ajson:
                qg_pair = {
                    "query": qg_pair['query'],
                    "answer": ajson["answer"],
                    "groundings": qg_pair['groundings']
                }
        elif self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            ans_system_prompt = "You are a helpful assistant, that given a query and list of groundings (citations related to the query), generates meaningful answer to the question."
            ans_prompt_tokens = self.tokenizer([get_prompt_token(ans_instruction_prompt, ans_system_prompt, self.tokenizer)], return_tensors = "pt", truncation = True, padding = True).to(self.device)
            aoutputs = execute_llama_LLM_task(self.llm, ans_prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            asummary = aoutputs[0]
            print(f'generated response for question: ', asummary)
            if "Input for your task" in asummary:
                ti = asummary.index("Input for your task")
                ajson = extract_json_text_by_key(asummary[ti:], "answer")
                print('qjson', ajson)
                if ajson != None and "answer" in ajson:
                    qg_pair = {
                        'query': qg_pair['query'],
                        'answer': ajson['answer'],
                        'groundings': qg_pair['groundings']
                    }
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            ans_system_prompt = "You are a helpful assistant, that given a query and list of groundings (citations related to the query), generates meaningful answer to the question."
            asummary = execute_llama_task_api(self.llm, ans_instruction_prompt, ans_system_prompt)
            print('generated response: ', asummary)
            ajson = extract_json_text_by_key(asummary, "answer")
            if ajson != None and "answer" in ajson:
                qg_pair = {
                    'query': qg_pair['query'],
                    'answer': ajson['answer'],
                    'groundings': qg_pair['groundings']
                }
        else:
            ans_system_prompt = "You are a helpful assistant, that given a query and list of groundings (citations related to the query), generates meaningful answer to the question."
            ans_prompt_tokens = [get_prompt_token(ans_prompt_text, ans_system_prompt, self.tokenizer) for ans_prompt_text in ans_instruction_prompts]
            aoutputs = execute_LLM_tasks(self.llm, ans_prompt_tokens, max_new_tokens=3000, temperature=0.6, top_p=0.9)
            asummary = aoutputs[0].outputs[0].text.strip()
            print(f'generated response for answer: ', asummary)
            ajson = extract_json_text_by_key(asummary, "answer")
            if ajson != None and "answer" in ajson:
                cleaned_answers = self.__cleanup_text([ajson["answer"]], 300)
                if cleaned_answers[0] != "":
                    qg_pair = {
                        "query": qg_pair['query'],
                        "answer": cleaned_answers[0],
                        "groundings": qg_pair['groundings']
                    }
                
        return qg_pair

    def set_filename(self, filename):
        self.filename = filename
        self.company_name = COMPANY_DICT[filename.split('_')[1]]
        
    def generate_answer(self, refine_answers = False):
        
        iquery_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_generated_queries.json'

        if os.path.exists(iquery_store_fp):
            with open(iquery_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of entities formed: ', len(query_arr))

            
            #chunk_topics = ",".join(chunk_obj["topics"])
            #random_indices = random.sample(range(0, len(all_factoids)), MAX_FACTOIDS_TO_SAMPLE)

            metadata = f'Company: {self.company_name} | SEC Filing: 10-K'
            print('\nStarting answer generation for batch of questions\n')
            sampled_entities = query_arr.keys()
            for entity in sampled_entities:
                filtered_queries = query_arr[entity]
                for qi, query_obj in enumerate(filtered_queries):
                    grounding_texts = [gr['text'] for gr in query_obj["groundings"]]
                    qng_pair = { 'query': query_obj['query'], "groundings": grounding_texts }
                    #groundings_arr_batch = [qs["groundings"] for qs in filtered_queries[i:(i+self.prompt_batch_size)]] 
                    #groundings_doc_batch = ["[" + ",".join(f"\"{item}\"" for item in groundings_arr) + "]" for groundings_arr in groundings_arr_batch]
                    qobj = self.__generate_answer(qg_pair=qng_pair, metadata=metadata, entity=entity)
                    if 'answer' in qobj:
                        filtered_queries[qi]['answer'] = qobj['answer']
                query_arr[entity] = filtered_queries                    

                if refine_answers:
                    pass

                query_store["queries"] = query_arr

                with open(iquery_store_fp, 'w') as fp:
                    json.dump(query_store, fp) 
        else:
            SystemExit('Chunk store not found!')

    def destroy(self):
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        os._exit(0)

if __name__ == "__main__":
    st = time()
    log_fp = f'logs/bu-answer-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--refine_answers', type = bool, default = False, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--prompt_batch_size', type = int, default = 1, required = False)

    args = parser.parse_args()

    ans_gen = AnswerGenerator(model_index = args.model_index, prompt_batch_size = args.prompt_batch_size)
    print(f'\n\nGenerating answers for file: {args.filename}')
    ans_gen.set_filename(args.filename)
    ans_gen.generate_answer(refine_answers = args.refine_answers)
    torch.cuda.empty_cache()

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()

    ans_gen.destroy()