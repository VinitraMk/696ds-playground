import os
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import multiprocessing
import numpy as np
import json
from time import time
import sys
import argparse
import re

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

MODEL_NAME = 'Qwen/QwQ-32B-AWQ'

class QueryGenerator:

    def __init__(self, filename, model_index = 0, topic_index = 0):
        self.filename = filename
        self.topic_index = topic_index
        self.company_abbr = filename.split('_')[1]
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
        self.model_folder = "qwq"

    def __extract_json_from_text(self, raw_text, target_key):
        """
        Searches raw text for a JSON object that contains a specific key
        and returns it as a Python dictionary. Returns None if not found.
        """
        # Match any JSON object containing the key: { "target_key": "some_value" }
        pattern = rf'\{{[^{{}}]*"{re.escape(target_key)}"\s*:\s*"[^"{{}}]*"[^{{}}]*\}}'

        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def __is_valid_sentence(self, sentence, word_count_limit = 100):
    
        sentence = sentence.strip()
        
        if re.search(r'[\u4e00-\u9fff]', sentence):  # Unicode range for Chinese characters
            return False
        
        if len(sentence) > 10 and " " not in sentence:  # remove strings with no spaces
            return False
        
        if re.search(r'_[a-zA-Z]', sentence):  # Detect underscores replacing spaces
            return False
        
        if len(sentence.split(" ")) > word_count_limit or len(sentence.split(" ")) < 7:
            return False
        
        return True
    
    def __execute_LLM_task_chain(self, prompt, max_new_tokens, temperature = 0.3, top_p = 0.9, return_prompt_messages = False, prev_messages = []):
        if len(prev_messages) == 0:
            return self.__execute_LLM_task(prompt = prompt, max_new_tokens = max_new_tokens, temperature = temperature, top_p = top_p, return_prompt_messages = return_prompt_messages)
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"./models/{self.model_name}")
            tokenizer.lang = "en"
            prev_messages.append({"role": "user", "content": prompt })
            text = tokenizer.apply_chat_template(
                prev_messages,
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
            if return_prompt_messages:
                return response_text, prev_messages
            return response_text

    
    def __execute_LLM_task(self, prompt, max_new_tokens, temperature = 0.3, top_p = 0.9, return_prompt_messages = False):
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
        if return_prompt_messages:
            return response_text, messages
        return response_text
    
    def __cleanup_query(self, text, keys):
        # Use regex to extract the JSON block
        match = re.search(r'\{[\s\S]*?\}', text)
        if not match:
            print("No JSON object found in the input.")
            return {}

        json_str = match.group(0)

        try:
            # Attempt to parse the JSON string
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON parsing failed: " + str(e))
            return {}

        # Safely extract the required fields
        kv0 = data.get(keys[0], "")
        kv1 = data.get(keys[1], "")
        return {keys[0]: kv0, keys[1]: kv1}

    def __extract_clean_queryset(self, gen_query_response):
        instruction_prompt = """
        ### Task:
        Extract and format the following sections from the provided text into a **strict Markdown structure**:

        ### Required Sections:  
        - **Query**: The question posed (beginning with "Query:").  

        ### Generation guidelines
        - Output **only** the desired markdown structure, no explanations.

        ### Formatting Rules:  
        - Clean all text, remove extra spaces, unnecessary or incorrect punctuations.
        - Remove trailing spaces, commas, hyphens.
        - Fix all obvious typos
        - Correct grammatical errors.

        ### Input Format:
        **Query:** <query text>

        ### Output Format:
        {
            "query": <query text>,
        }

        ### Example Input:  
        **Query:** Why adjust_[inventory values=_ quarterly?  

        ### Example Output: 
        {
            "query": "Why adjust inventory values quarterly?",
        } 

        ### Task Input
        """

        instruction_prompt = instruction_prompt + f"\n**Text:** {gen_query_response}"
        summary = self.__execute_LLM_task(instruction_prompt, max_new_tokens=3000, temperature=0.1, top_p = 0.9)
        print('\nGenerated structured Query:\n', summary)
        query_dict = self.__extract_json_from_text(summary)
        print('query dict', query_dict)
        '''
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
        '''

    def __generate_answer(self, fact_doc_text, metadata, query):

        instruction_prompt = """
        Analyze the provided set of factoids, the metadata and the query give answera to the question from the provided factoids.

        ### Desired response structure:  
        - **Answer**: Write a concise answer to the question using the provided factoids.

        ### Generation Rules
        - **Do not use chinese characters** in your response.
        - Phrase your response as concisely as possible.
        - Keep the answer under 300 tokens.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - End the response with `###EOF###`.  
        - Use the example structure to return the final response.
        - Label your chain-of-thought as `###COT###` and the final answer as `###FINALRESPONSE###`.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        **Metadata:** <meta data of the main company upon which the factoids are based.>
        **Factoids:** [\<list of factoids\>]
        **Query:** <query text>

        ### Output format:
        **Answer:** <answer to the query in input>

        ### Example Output:

        ###FINAL RESPONSE###
        **Answer:**
        Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
        On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation.
        From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments.

        ### Input for your task:
        """

    def __generate_grounding_reasoning_pair(self, fact_doc_text, metadata, qna_dict):
        grounding_instruction_prompt = """
        ### Task:
        Analyze the provided set of factoids, the metadata, question & answer pair and generate groundings for the question and answer pair.
        Groundings are factoids that support the answer to the question.

        ### Generation Rules
        - **Do not use chinese characters** in your response.
        - Phrase your response as concisely as possible.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - End the response with `###EOF###`.  
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.
        - The groundings should a list of factoids that are picked directly from the input factoids. **Do not** generate new factoids to support the answer.

        ### Input format:
        **Metadata:** <meta data of the main company upon which the factoids are based.>
        **Factoids:** [\<list of factoids\>]
        **Question:** <question text>
        **Answer:** <answer text>

        ### Output format:
        Groundings: [\<list of factoids picked supporting the answer\>]

        ### Example Output (JSON):
        Groundings: {
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
        }
        
        ### Input for your task:
        """
        grounding_instruction_prompt = grounding_instruction_prompt + f"\n**Metadata:** {metadata}\n**Factoids:**{fact_doc_text}\n**Question:** {qna_dict['query']}\n**Answer:** {qna_dict['answer']}"

        summary, messages = self.__execute_LLM_task_chain(grounding_instruction_prompt, max_new_tokens=5000, temperature=0.2, top_p = 0.9, return_prompt_messages = True)
        print('Generated Response for groundings:\n', summary)
        return {}


    def __generate_query_answer_pair(self, fact_doc_text, metadata):

        qstn_instruction_prompt = """
        ### Task:
        Given the list of factoids below and metadata, generate a complex question that requires reasoning over multiple factoids

        ### Generation Rules
        - **Do not use chinese characters** in your response.
        - Keep the generated query under 100 tokens.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - End the response with `###EOF###`.
        - Don't think for more than 2000 tokens
        - Label your chain of thoughts or reasoning as ### COT ### and final response as ### FINAL RESPONSE ###
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]

        ### Output format:
        Query: <question generated from fact(s) in the given text document>

        ### Example Input
        Metadata: Company name: AAPL | SEC Filing: 10-K
        Factoids: ["Apple committed to carbon neutrality across its supply chain by 2030.",
            "Apple sources renewable energy for its global operations.",
            "Apple integrates recycled materials into product design.",
            "The company works with suppliers to reduce emissions.",
            "Upfront costs have increased due to sustainability investments."
        ]

        ### Example Output:
        Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?",

        ### Input for your task:
        """

        answer_instruction_prompt = """
        ### Task:
        Given the list of factoids below, metadata, and a query, summarize and return an answer to the query using the given factoids.

        ### Answer Generation Rules
        - Generated answer is ONLY from the given factoids, **do not** hallucinate new information or factoids to answer the query.
        - Return the final answer as one single concise paragraph in a json object. Use the example output as reference for structure.
        - **Do not put chinese characters** in your response.
        - Keep the generated answer concise and under 500 words.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response.
        - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]
        Query: <query text>

        ### Output format (JSON):
        Answer: {
            "answer": <answer to the question, generated from fact(s) in the given text document>
        }

        ### Example Input
        Metadata: Company name: AAPL | SEC Filing: 10-K
        Factoids: ["Apple committed to carbon neutrality across its supply chain by 2030.",
            "Apple sources renewable energy for its global operations.",
            "Apple integrates recycled materials into product design.",
            "The company works with suppliers to reduce emissions.",
            "Upfront costs have increased due to sustainability investments."
        ]
        Query: "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?"

        ### Example Output:
        Answer: {
            "answer": "Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs. On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation. From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments."
        }

        ### Input for your task:
        """

        qstn_instruction_prompt = qstn_instruction_prompt + f"\nMetadata: {metadata}\nFactoids: {fact_doc_text}"
        qna_pair = {}

        summary, messages = self.__execute_LLM_task_chain(qstn_instruction_prompt, max_new_tokens=3000, temperature=0.2, top_p = 0.9, return_prompt_messages = True)
        print('Generated Response:\n', summary)
        if "Query:" in summary:
            qi = summary.index("Query:")
            if "\n" in summary[(qi+6):]:
                nli = summary.index("\n")
                query_str = summary[(qi+6):nli].strip()
                print('Generated query: ', query_str)
            if query_str != None or query_str != "":
                messages.append({"role": "system", "content": query_str })
                answer_instruction_prompt = answer_instruction_prompt + f"\nMetadata: {metadata}\nFactoids: {fact_doc_text}\nQuery: {query_str}"
                ans_summary, ans_messages = self.__execute_LLM_task_chain(answer_instruction_prompt, max_new_tokens=4096, temperature=0.2, top_p = 0.9,
                    return_prompt_messages = True)
                print('Generated answer response:\n', ans_summary)
                answer_json = self.__extract_json_from_text(ans_summary, "answer")
                if answer_json != None:
                    qna_pair = {
                        'query': query_str,
                        'answer': answer_json["answer"]
                    }
                    return qna_pair
        return None


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
        - Keep the query under 100 tokens.
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

        ### Example Output:

        ###FINAL RESPONSE###
        Query: How does Apple’s commitment to achieving carbon neutrality across its supply chain and
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

        summary = self.__execute_LLM_task(instruction_prompt, max_new_tokens=5000, temperature=0.3, top_p = 0.5)
        print('Generated Response:\n', summary)
        query_dict = self.__cleanup_query(summary)
        print(query_dict)
        #self.__extract_clean_queryset(summary)

    def generate_query(self, no_of_trials = 10):
        st = time()

        log_fp = f'logs/bu-query-logs.txt'
        log_file = open(log_fp, 'w')
        old_stdout = sys.stdout
        sys.stdout = log_file

        # Filter relevant chunks

        all_resp = []
        #no_of_trials = 10

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
            print('total length of filtered array: ', len(filtered_factoids))
            #factoid_subarr = all_factoids[:MAX_FACTOIDS_TO_SAMPLE]
            metadata = f'Company: {self.company_abbr} | SEC Filing: 10-K'
            #for idx in random_indices:
                #factoid_subarr.append(all_factoids[idx])
            all_resp = []
            print('\nStarting query generation for batch of factoids\n')
            for i in range(0, len(filtered_factoids), MAX_FACTOIDS_TO_SAMPLE):
                factoid_subarr = filtered_factoids[i:i+MAX_FACTOIDS_TO_SAMPLE]
                if len(factoid_subarr) >= 15:
                    factoid_str = "[" + ",".join(f"{item['factoid']}" for item in factoid_subarr) + "]"
                    qna_pairs = []
                    print(f'\nRunning {no_of_trials} qna generation for factoids batch {i}')
                    for i in range(no_of_trials):
                        query_dict = self.__generate_query_answer_pair(factoid_str, metadata)
                        print('generated qna pair: ', query_dict)
                        if query_dict != None and self.__is_valid_sentence(query_dict["query"]) and self.__is_valid_sentence(query_dict["answer"], 500):
                            qna_pairs.append(query_dict)
                    print(f'Found {len(qna_pairs)} valid qna pairs')
                
                '''

                for qnapair in qna_pairs:
                    query_dict = self.__generate_grounding_reasoning_pair(factoid_str, metadata, qnapair)
                    if "reasoning" in query_dict and "groundings" in query_dict:
                        query_dict = query_dict | qnapair
                        all_resp.append(query_dict)
                '''
                all_resp = all_resp + qna_pairs
            print('No of valid whole set: ', len(all_resp))
            query_json_path = f'./data/queries/{self.model_folder}/{self.filename}_gen_queries.json'
            if os.path.exists(query_json_path):
                with open(query_json_path, 'r') as fp:
                    queries = json.load(fp)
            else:
                queries = { 'queries': [] }
            topic_queries = [tq for tq in queries["queries"] if tq["topic"] == SEED_METADATA_TOPICS[self.topic_index]]
            #print("topic queries", topic_queries, len(topic_queries), query_dict)
            if len(topic_queries) > 0:
                topic_queries = topic_queries[0]
                topic_queries["query_sets"] = topic_queries["query_sets"] + all_resp
                for iq,_ in enumerate(queries["queries"]):
                    if queries["queries"][iq]["topic"] == SEED_METADATA_TOPICS[self.topic_index]:
                        queries["queries"][iq]["query_sets"] = topic_queries["query_sets"]
            else:
                topic_queries = { "topic": SEED_METADATA_TOPICS[self.topic_index], "query_sets": [] }
                topic_queries["query_sets"] = all_resp
                queries["queries"].append(topic_queries)

            with open(f'./data/queries/{self.model_folder}/{self.filename}_gen_queries.json', 'w') as fp:
                json.dump(queries, fp) 
        else:
            print('Chunk store not found!')
            SystemExit()

        print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
        sys.stdout = old_stdout
        log_file.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--no_of_trials', type = int, default = 10, required = False)

    args = parser.parse_args()

    #filename = '10-K_AMD_20231230'
    filename = '10-K_NVDA_20240128'

    query_gen = QueryGenerator(filename, model_index = args.model_index, topic_index = args.topic_index)
    query_gen.generate_query(no_of_trials = args.no_of_trials)
