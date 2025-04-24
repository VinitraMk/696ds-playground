import os
import torch
from vllm import LLM
import multiprocessing
import json
from time import time
import sys
import argparse
import re

from utils.string_utils import is_valid_sentence, extract_json_text_by_key
from utils.llm_utils import get_prompt_token, execute_LLM_tasks

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
PROMPT_BATCH_SIZE = 3
NO_OF_TRIALS = 5

class AnswerGenerator:

    def __init__(self, filename, model_index = 6):
        self.filename = filename
        self.company_abbr = COMPANY_DICT[filename.split('_')[1]]
        self.model_name = MODELS[model_index]
        if "QwQ" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                    quantization = "awq",
                    download_dir = HF_CACHE_DIR,
                    max_model_len = 2048 * 4,
                    #gpu_memory_utilization=0.95,
                    tensor_parallel_size=torch.cuda.device_count())
        elif "Qwen2.5" in self.model_name:
            self.llm = LLM(model=f"./models/{self.model_name}",
                quantization = "gptq_marlin",
                download_dir = HF_CACHE_DIR,
                max_model_len = 2048 * 4,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count())
        elif "Llama" in self.model_name:
            #mf = self.model_name.split("/")[1]
            self.__quantize_llama(self.model_name)
            self.llm = LLM(model=f"./models/llama/{self.model_name}-awq",
                quantization="awq",
                gpu_memory_utilization=0.8,
                download_dir = f'./models/llama/{self.model_name}-awq',
                tensor_parallel_size=torch.cuda.device_count())
            self.model_folder = "llama"
        self.model_folder = "qwq"


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
        cleanup_system_prompt = "You are a helpful assistant, that given a text returns a cleaned, grammatically meaningful sentence if it exists."
        cleanup_instruction_prompt_tokens = [get_prompt_token(prompt, cleanup_system_prompt, self.model_name) for prompt in cleanup_instructions]
        coutputs = execute_LLM_tasks(self.llm, prompts=cleanup_instruction_prompt_tokens, max_new_tokens=2000, temperature=0.1, top_p=0.9)
        for o in coutputs:
            if o:
                osummary = o.outputs[0].text.strip()
            else:
                print(o)
                osummary = ""
            print('generated clean up response', osummary)
            match = re.search(r'Cleaned\s+Text:\s*"([^"]+)"', osummary)
            if (match and is_valid_sentence(match.group(1), wc)):
                cleaned_sentences.append(match.group(1))
            else:
                cleaned_sentences.append("")
        return cleaned_sentences

    def __generate_answers(self, fact_doc_texts, factoids_arr_batch, query_strs, metadata):

        ans_instruction_prompt = """
        ### Task:
        Given a query, some metadata and a list of factoids, summarize and return an answer to the query using the given factoids.

        ### Answer Generation Rules
        - **No opinions, adjectives, elaborations or extra details**
        - Generated answer is ONLY from the given factoids, **do not** hallucinate new information or factoids to answer the query.
        - Return the final answer as one single concise paragraph in a json object. Use the example output as reference for structure.
        - **Do not put chinese characters** in your response. Return responses only in English.
        - Keep the generated answer concise and under 200 words.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response.
        - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
        - **Do not copy example from the prompt** in your response.
        - Don't think for more than 3000 tokens.

        ### Input format:
        Query: <query text>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]

        ### Output format (JSON):
        Answer: {
            "answer": <answer to the question, generated from fact(s) in the given text document>
        }

        ### Example Input
        Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?"
        Metadata: Company name: AAPL | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Factoids: ["Apple committed to carbon neutrality across its supply chain by 2030.",
            "Apple sources renewable energy for its global operations.",
            "Apple integrates recycled materials into product design.",
            "The company works with suppliers to reduce emissions.",
            "Upfront costs have increased due to sustainability investments."
        ]

        ### Example Output:
        Answer: {
            "answer": "Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs. On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation. From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments."
        }

        ### Input for your task:
        """

        zipped_qsnts_factoids = list(zip(query_strs, fact_doc_texts))
        ans_instruction_prompts = [ans_instruction_prompt + f"\nQuery: {qo[0]}\nMetadata: {metadata}\nFactoids: {qo[1]}" for qo in zipped_qsnts_factoids]
        ans_system_prompt = "You are a helpful assistant, that given a list of factoids and a query, generates meaningful answer to the question based on the factoids."
        ans_prompt_tokens = [get_prompt_token(ans_prompt_text, ans_system_prompt, self.model_name) for ans_prompt_text in ans_instruction_prompts]
        aoutputs = execute_LLM_tasks(self.llm, ans_prompt_tokens, max_new_tokens=3000, temperature=0.6, top_p=0.9)

        #summary, qmessages = self.__execute_LLM_task_chain(qstn_instruction_prompt, max_new_tokens=3000, temperature=0.2, top_p = 0.9, return_prompt_messages = True)
        #print('Generated Response:\n', summary)
        qna_pairs = []
        zipped_qsnts_factoids = list(zip(query_strs, factoids_arr_batch))
        missed_qstns = []
        for zqf, o in zip(zipped_qsnts_factoids, aoutputs):
            asummary = o.outputs[0].text.strip()
            print(f'generated response for answer: ', asummary)
            ajson = extract_json_text_by_key(asummary, "answer")
            if ajson != None and "answer" in ajson:
                cleaned_answers = self.__cleanup_text([ajson["answer"]], 200)
                if cleaned_answers[0] != "":
                    qna_pairs.append({
                        "query": zqf[0],
                        "answer": cleaned_answers[0],
                        "factoids": zqf[1]
                    })
                else:
                    missed_qstns.append({
                        'query': zqf[0],
                        'factoids': zqf[1]
                    })
            else:
                missed_qstns.append({
                    'query': zqf[0],
                    'factoids': zqf[1]
                })

        print('no of valid qna pairs', len(qna_pairs))
        print('no of invalid qna pairs', len(missed_qstns))

        return qna_pairs, missed_qstns
        
    def generate_answer(self, topic_index = 0):
        

        # Filter relevant chunks

        all_resp = []
        #no_of_trials = 10

        query_store_fp = f'intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
        
        #chunk_obj = [chunk for chunk in chunk_store["chunks"] if chunk["chunk_filename"] == chunk_fn]
        if os.path.exists(query_store_fp):
            with open(query_store_fp, 'r') as fp:
                query_store = json.load(fp)
            query_arr = query_store["queries"]
            print('total no of queries formed: ', len(query_arr))
            #chunk_topics = ",".join(chunk_obj["topics"])
            #random_indices = random.sample(range(0, len(all_factoids)), MAX_FACTOIDS_TO_SAMPLE)
            filtered_queries = [querysets for querysets in query_arr if querysets["topic"] == SEED_METADATA_TOPICS[topic_index]]
            if len(filtered_queries) > 0:
                filtered_queries = filtered_queries[0]["query_sets"]
            else:
                print(f'no queries formed for the topic: {SEED_METADATA_TOPICS[topic_index]}')
                SystemExit()

            print('total length of filtered array: ', len(filtered_queries))
            #factoid_subarr = all_factoids[:MAX_FACTOIDS_TO_SAMPLE]
            metadata = f'Company: {self.company_abbr} | SEC Filing: 10-K | Related topic: {SEED_METADATA_TOPICS[topic_index]}'
            #for idx in random_indices:
                #factoid_subarr.append(all_factoids[idx])
            all_resp = []
            print('\nStarting answer generation for batch of factoids\n')
            for bi,i in enumerate(range(0, len(filtered_queries), PROMPT_BATCH_SIZE)):
                query_strs_batch = [qs["query"] for qs in filtered_queries[i:(i+PROMPT_BATCH_SIZE)]]
                factoids_arr_batch = [qs["factoids"] for qs in filtered_queries[i:(i+PROMPT_BATCH_SIZE)]] 
                factoids_doc_batch = ["[" + ",".join(f"\"{item['factoid']}\"" for item in factoid_subarr) + "]" for factoid_subarr in factoids_arr_batch]
                print(f'\nRunning answer generation for factoids batch {bi}')
                qobjs, missed_qstns = self.__generate_answers(factoids_doc_batch, factoids_arr_batch, query_strs_batch, metadata)
                all_resp.extend(qobjs)
                attempts = 0
                while (len(missed_qstns) != 0) and (attempts < NO_OF_TRIALS):
                    query_strs_batch = [qs["query"] for qs in missed_qstns]
                    factoids_arr_batch = [qs["factoids"] for qs in missed_qstns] 
                    factoids_doc_batch = ["[" + ",".join(f"\"{item['factoid']}\"" for item in qs['factoids']) + "]" for qs in missed_qstns]
                    qobjs, missed_qstns = self.__generate_answers(factoids_doc_batch, factoids_arr_batch, query_strs_batch, metadata)
                    all_resp.extend(qobjs)
                    attempts+=1

            print('No of valid qna pairs formed so far: ', len(all_resp))
            if len(all_resp) > 0:
                query_json_path = f'./intermediate_data/query_sets/{self.model_folder}/{self.filename}_gen_queries.json'
                queries = { 'queries': [] }
                #print("topic queries", topic_queries, len(topic_queries), query_dict)
                topic_queries = { "topic": SEED_METADATA_TOPICS[topic_index], "query_sets": [] }
                topic_queries["query_sets"] = all_resp
                queries["queries"].append(topic_queries)

                with open(query_json_path, 'w') as fp:
                    json.dump(queries, fp) 
        else:
            print('Chunk store not found!')
            SystemExit()

if __name__ == "__main__":
    st = time()
    log_fp = f'logs/bu-answer-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    ans_gen = AnswerGenerator(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(len(SEED_METADATA_TOPICS)):
            print(f'Generating answers for questions on topic: {SEED_METADATA_TOPICS[ti]}')
            ans_gen.generate_answer(topic_index = ti)
            print(f'Finished generating answers for topic: {SEED_METADATA_TOPICS[ti]}')
    else:
        print(f'Generating questions for topic: {SEED_METADATA_TOPICS[args.topic_index]}')
        ans_gen.generate_answer(topic_index = args.topic_index)
        print(f'Finished generating answers for topic: {SEED_METADATA_TOPICS[args.topic_index]}')

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
