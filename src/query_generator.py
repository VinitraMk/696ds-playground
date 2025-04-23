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
#from awq import AutoAWQForCausalLM, AWQConfig
import random

COMPANY_DICT = {
    'INTC': 'Intel Corp.',
    'AMD': 'AMD Inc.',
    'NVDA': 'Nvidia Corp.',
    'TSLA': 'Tesla Inc.'
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

MODEL_NAME = 'Qwen/QwQ-32B-AWQ'

class QueryGenerator:

    def __init__(self, filename, model_index = 0, topic_index = 0):
        self.filename = filename
        self.topic_index = topic_index
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

    # utility functions

    def __quantize_llama(self, model_name):
        quant_model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            quant_config=AWQConfig(
                bits=4,
                group_size=128,  # Optional; depends on the model you're targeting
                zero_point=True, # Optional; enables zero-point quantization
            ),
            cache_dir = HF_CACHE_DIR
        )
        quant_model.save_pretrained(f"./models/llama/{model_name}-awq")

    def __extract_json_text_by_key(self, raw_text, target_key):
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
    
    def __extract_json_array_by_key(self, raw_text, target_key):
        """
        Extracts a list of strings from a JSON-like structure in LLM output
        where the given key maps to a string array.

        Returns a list of strings, or None if not found or not valid.
        """
        # Regex to find: "key": [ "value1", "value2", ... ]
        pattern = rf'"{re.escape(target_key)}"\s*:\s*\[(?:\s*"[^"]*"\s*,?\s*)+\]'

        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            json_fragment = "{" + match.group(0) + "}"
            try:
                parsed = json.loads(json_fragment)
                return parsed[target_key]
            except json.JSONDecodeError:
                return None
        return None

    def __is_valid_sentence(self, sentence, word_count_limit = 100):
    
        sentence = sentence.strip()

        if sentence == 'None' or sentence == "":
            return False
        
        if re.search(r'[\u4e00-\u9fff]', sentence):  # Unicode range for Chinese characters
            return False
        
        if len(sentence) > 10 and " " not in sentence:  # remove strings with no spaces
            return False
        
        if re.search(r'_[a-zA-Z]', sentence):  # Detect underscores replacing spaces
            return False
        
        if len(sentence.split(" ")) > word_count_limit or len(sentence.split(" ")) < 7:
            return False
        
        return True
    

    # utility LLM functions

    def __get_prompt_token(self, prompt_text):
        
        tokenizer = AutoTokenizer.from_pretrained(f"./models/{self.model_name}")
        tokenizer.lang = "en"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts factoids from text."},
            {"role": "user", "content": prompt_text}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    
    def __execute_LLM_task_chain(self, prompts, max_new_tokens, temperature = 0.3, top_p = 0.9, return_prompt_messages = False, prev_messages = []):
        if len(prev_messages) == 0:
            return self.__execute_LLM_task(prompt = prompts, max_new_tokens = max_new_tokens, temperature = temperature, top_p = top_p, return_prompt_messages = return_prompt_messages)
        else:
            #tokenizer = AutoTokenizer.from_pretrained(f"./models/{self.model_name}")
            #tokenizer.lang = "en"
            prev_messages.append({"role": "user", "content": prompt })
            #text = tokenizer.apply_chat_template(
                #prev_messages,
                #tokenize=False,
                #add_generation_prompt=True
            #)
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.5,
                stop=["###EOF###"]
            )
            #model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
            outputs = self.llm.generate(prompts, sampling_params)
            return outputs
            #response_text = tokenizer.decode(
                #outputs[0].outputs[0].token_ids,
                #skip_special_tokens=True,
                #clean_up_tokenization_spaces=True  # Ensures proper spacing
            #)
            #return outputs[0].outputs[0].text if outputs else ""
            if return_prompt_messages:
                return response_text, prev_messages
            return response_text

    
    def __execute_LLM_tasks(self, prompts, max_new_tokens, temperature = 0.3, top_p = 0.9):
        #tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir = os.environ['HF_HOME'])
        #tokenizer = AutoTokenizer.from_pretrained(f"./models/{self.model_name}")
        #tokenizer.lang = "en"
        #messages = [
            #{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that given a task returns a response following the exact structured output format specified in the prompt. Respond only in English"},
            #{"role": "user", "content": prompt}
        #]
        #text = tokenizer.apply_chat_template(
            #messages,
            #tokenize=False,
            #add_generation_prompt=True
        #)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.5,
            stop=["###EOF###"]
        )
        #model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
        outputs = self.llm.generate(prompts, sampling_params)
        #response_text = tokenizer.decode(
            #outputs[0].outputs[0].token_ids,
            #skip_special_tokens=True,
            #clean_up_tokenization_spaces=True  # Ensures proper spacing
        #)
        #return outputs[0].outputs[0].text if outputs else ""
        return outputs
    
    def __cleanup_sentence(self, sentences):
        cleanup_instruction = """
        ### Task:
        Given a sentence, riddled with grammatical, punctuation, insensible/gibberish words or typos (missing spaces), remove
        the errors and return a clean, grammatically correct and meaningful sentence if it exists. If a meaningful sentence doesn't exist,
        return a blank string "". If there are no errors in the text, return it as is.

        ### Cleanup Rules:
        - Remove punctuation mistakes such as missing spaces, incorrectly placed hyphens, obvious typos, etc.
        - Return a grammatically correct and meaningful sentence. If it doesn't exist, return "".
        - Remove chinese characters.
        - If the sentence contains gibberish, insensible words return "".
        - **Do not** make a completely new sentence.
        - **Do not** hallucinate completely new words in the sentence.
        - **Do not** put chinese words in the sentence.

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
        Cleaned Text: "Apple works with suppliers to transition to clean energy and energy-efficient production methods."

        ### Input for your task:
        """
        cleaned_sentences = []
        cleanup_instructions = [cleanup_instruction + f"\nText: \"{sentence}\"" for sentence in sentences]
        cleanup_instruction_prompt_tokens = [self.__get_prompt_token(prompt) for prompt in cleanup_instructions]
        coutputs = self.__execute_LLM_tasks(prompts=cleanup_instruction_prompt_tokens, max_new_tokens=2000, temperature=0.1, top_p=0.9)
        for o in coutputs:
            osummary = o.outputs[0].text.strip()
            print('generated clean up response', osummary)
            match = re.search(r'Cleaned\s+Text:\s*"([^"]+)"', osummary)
            if (match and self.__is_valid_sentence(match.group(1))):
                cleaned_sentences.append(match.group(1))
            else:
                cleaned_sentences.append("")
        return cleaned_sentences
    
    def __clean_json_array(self, json_arr):
        cleaned_sentences = []
        if json_arr != None and len(json_arr) > 0:
            cleaned_sentences = self.__cleanup_sentence(json_arr)
        return cleaned_sentences

    def __generate_grounding_reasoning_pair(self, fact_doc_text, metadata, qna_dict, no_of_trials = 10):
        grounding_instruction_prompt = """
        ### Task:
        Analyze the provided question and answer pair, set of factoids and the metadata about the facts, generate groundings for the question and answer pair.
        Groundings are factoids that support the answer to the question. The factoids don't have to directly support the answer but should help indirectly answering the
        provided question.

        ### Generation Rules
        - The groundings should be factoids picked directly from the provided factoids in the input prompt.
        - **Do not** generate new factoids to put in the groundings to support the answer.
        - Use the example structure as reference to return the final response.
        - Return clean groundings with no typos, grammatical mistakes or erronuous punctuations.
        - **Do not use chinese characters** in your response.
        - Phrase your response as concisely as possible.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - **Do not copy example from the prompt** in your response.
        - **Don't think** for more than 4000 tokens

        ### Input format:
        Question: <question text>
        Answer: <answer text>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]

        ### Output format:
        "groundings": [\<list of factoids picked supporting the answer\>]

        ### Example Input:
        Question: How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?
        Answer: Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
        Metdata: Company: Apple | SEC-filing: 10-K | Related Topic: Risk Factors and Challenges
        Factoids: [
            "Apple has committed to achieving carbon neutrality across its entire business, including the supply chain and product life cycle, by 2030.",
            "The company has made significant investments in renewable energy and low-carbon manufacturing technologies.",
            "Apple requires its suppliers to adhere to strict environmental standards, including reducing carbon emissions and using renewable electricity.",
            "Apple’s Supplier Clean Energy Program has contributed to its goal of reducing emissions across the supply chain.",
            "The company reports that its use of recycled and sustainable materials in products is increasing year-over-year.",
            "Apple continues to invest in energy-efficient technologies that improve operational efficiency across its facilities.",
            "The company discusses potential risks in its 10-K filings related to environmental regulation and climate change, including potential costs from future carbon taxes.",
            "Apple acknowledges short-term cost increases related to sustainability efforts but positions them as long-term strategic investments.",
            "The company’s ESG initiatives are framed as important for protecting brand value and aligning with consumer expectations.",
            "Environmental sustainability is identified as a strategic area for long-term growth and innovation in Apple’s regulatory and investor communications.",
            "Apple’s financial risk disclosures related to foreign exchange fluctuations in emerging markets.",
            "Details on litigation or legal contingencies unrelated to environmental policies.",
            "Statements about Apple’s R&D expenditures for chip architecture and performance optimization.",
            "Inventory management practices tied to consumer demand and holiday cycles.",
            "Tax strategies related to international operations and intellectual property.",
            "Descriptions of supply chain risks unrelated to ESG, such as natural disasters or geopolitical tension.",
            "Information about share repurchase programs or dividend policies.",
            "Macroeconomic risks including inflation or interest rate sensitivity."
        ]

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

        reasoning_instruction_prompt = """
        ### Task:
        Analyze the provided question and answer pair, set of groundings that support the answer and the metadata about the groundings, generate reasoning for each of the groundings.
        Reasoning is explanation of how or why the grounding sentence supports the answer to the question.

        ### Generation Rules
        - The every reasoning should be concise and kept under 50 words.
        - Use the example structure as reference to return the final response.
        - Return clean reasonings with no typos, grammatical mistakes or erronuous punctuations.
        - **Do not use chinese characters** in your response.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - **Do not copy example from the prompt** in your response.
        - **Don't think** for more than 4000 tokens

        ### Input format:
        Question: <question text>
        Answer: <answer text>
        Metadata: <meta data of the main company upon which the factoids are based.>
        Groundings: [\<list of factoids that ground or support the answer\>]

        ### Output format:
        "reasonings": [\<explanation about every grounding regarding how or why it supports the answer\>]

        ### Example Input:
        Question: How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?
        Answer: Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
        Metdata: Company: Apple | SEC-filing: 10-K | Related Topic: Risk Factors and Challenges
        Groundings: [
            "Apple has committed to achieving carbon neutrality across its entire business, including supply chain and product life cycle, by 2030.",
            "Apple invests in renewable energy and low-carbon manufacturing processes as part of its environmental sustainability goals.",
            "The company requires its suppliers to comply with its environmental standards, including carbon reduction initiatives.",
            "Apple works with suppliers to transition to clean energy and energy-efficient production methods.",
            "The company integrates recycled and sustainable materials into product design and manufacturing.",
            "Apple acknowledges that its environmental initiatives may lead to higher costs in the short term due to increased material and compliance expenses.",
            "The company anticipates that its ESG efforts will improve brand reputation and customer loyalty.",
            "Apple views its leadership in ESG initiatives as a competitive advantage, positioning it to mitigate future regulatory and environmental risks."
        ]

        ### Example Output (JSON):
        "reasonings": [
            "This grounding sets the foundation of the answer, as it directly reflects the core commitment Apple made—carbon neutrality across its business by 2030. This long-term strategic goal drives the operational, financial, and supplier-level changes discussed in the answer.",
            "Investing in renewable energy and low-carbon manufacturing directly increases Apple's upfront costs, which explains the immediate impact on its cost structure as noted in the answer.",
            "Requiring suppliers to comply with environmental standards introduces compliance costs and complexity across the supply chain, reinforcing the answer’s point about rising supplier-related expenses and risks.",
            "Collaborating with suppliers on clean energy adoption supports long-term operational efficiencies and cost savings, aligning with the answer's assertion that energy-efficient practices may reduce future operational costs.",
            "The use of recycled and sustainable materials increases product development costs due to design and processing complexity, which directly supports the answer’s explanation of how sustainability initiatives raise short-term expenses.",
            "This grounding validates the answer’s claim that ESG initiatives increase short-term costs, particularly due to the need for new materials and compliance requirements, thus reinforcing the financial trade-offs involved.",
            "Improved brand reputation and customer loyalty contribute to long-term profitability, as mentioned in the answer, especially among eco-conscious consumers willing to pay premium prices.",
            "By viewing ESG leadership as a competitive advantage, Apple positions itself to navigate future regulatory and environmental challenges, which supports the answer’s framing of long-term strategic benefits and risk mitigation."
        ]
        
        ### Input for your task:
        """
        grounding_instruction_prompts = [grounding_instruction_prompt + f"\nQuestion: {qna_dict['query']}\nAnswer: {qna_dict['answer']}\nMetadata: {metadata}\nFactoids: {fact_doc_text}" for _ in range(no_of_trials)]
        grounding_instruction_prompt_tokens = [self.__get_prompt_token(prompt) for prompt in grounding_instruction_prompts]
        goutputs = self.__execute_LLM_tasks(grounding_instruction_prompt_tokens, max_new_tokens=8192, temperature=0.1, top_p=0.9)

        #summary, gmessages = self.__execute_LLM_task(grounding_instruction_prompt, max_new_tokens=8192, temperature=0.1, top_p = 0.9, return_prompt_messages = True)
        
        #print('Generated Response for groundings:\n', summary)
        grounding_jsons = []
        ml = -1
        best_groundings = []
        for o in goutputs:
            gsummary = o.outputs[0].text.strip()
            groundings_json_arr = self.__extract_json_array_by_key(gsummary, "groundings")
            cleaned_groundings = self.__clean_json_array(groundings_json_arr)
            cleaned_groundings = [s for s in cleaned_groundings if s != ""]
            print('cleaned groundings: ', cleaned_groundings)
            ml = max(ml, len(cleaned_groundings))
            grounding_jsons.append(cleaned_groundings)
        
        print('extracted json', groundings_json_arr)
        best_groundings = [gs for gs in grounding_jsons if len(gs) == ml]
        best_grounding_strs = ["[" + ",\n".join(f"{item}" for item in cg) + "]" for cg in best_groundings]
        reasoning_instruction_prompts = [reasoning_instruction_prompt + f"\nQuestion: {qna_dict['query']}\nAnswer: {qna_dict['answer']}\nMetadata: {metadata}\nGroundings: {grounding_str}" for grounding_str in best_grounding_strs]
        reasoning_instruction_prompt_tokens = [self.__get_prompt_token(prompt) for prompt in reasoning_instruction_prompts]
        routputs = self.__execute_LLM_tasks(reasoning_instruction_prompt_tokens, max_new_tokens = 8192, temperature = 0.1, top_p = 0.9)

        # return only valid and the best reasoning-groundings pair
        ml = -1
        mli = []
        valid_groundings_coll = []
        valid_reasonings_coll = []
        for j,o in enumerate(routputs):
            rsummary = o.outputs[0].text.strip()
            print('Generated response for reasonings: ', rsummary)
            reasonings_json_arr = self.__extract_json_array_by_key(rsummary, "reasonings")
            print('extracted reasonings json', reasonings_json_arr)
            cleaned_reasonings = self.__clean_json_array(reasonings_json_arr)
            cleaned_groundings = best_groundings[j]
            valid_reasonings = [cr for _,cr in enumerate(cleaned_reasonings) if cr != ""]
            valid_groundings = [cleaned_groundings[ri] for ri,cr in enumerate(cleaned_reasonings) if cr != ""]
            if len(valid_groundings) > 0 and len(valid_reasonings) > 0:
                valid_groundings_coll.append(valid_groundings)
                valid_reasonings_coll.append(valid_reasonings)
        for j in range(len(valid_groundings_coll)):
            ml = max(ml, len(valid_groundings_coll[j]))
            if len(valid_groundings_coll[j]) == ml:
                mli.append(j)
        if len(mli) > 0:
            mi = random.choice(mli)
            return {
                'reasonings': valid_reasonings_coll[mi],
                'groundings': valid_groundings_coll[mi]
            }

        return None

    def __generate_grounding_reasoning_pair_iteratively(self, metadata, qna_dict, factoid_list, qna_messages):
        instruction_prompt = """
        ### Task:
        Analyze the provided set of question & answer pair, the provided factoid and metdata about the company upon which factoid is based,
        and return yes/no response whether the fact supports the answer to the question. Also state your reasoning for it, citing relevant part of the
        sentence in answer.

        ### Generation Rules
        - Return the final response in a structured JSON format, as shown in example. Use the example structure as reference to return the final response.
        - Keep the reasoning explanation concise and under 30 words.
        - **Do not use chinese characters** in your response for reasoning.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for reasoning.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        **Question:** <question text>
        **Answer:** <answer text>
        **Metadata:** <meta data of the main company upon which the factoids are based.>
        **Factoid:** <factoid text>

        ### Output format:
        Response: {
            "support": <yes/no response of whether the factoid supports the answer>
            "reasoning": <concise explanation of why factoid supports the answer, citing the relevant part of the answer>
        }

        ### Example Input:
        **Question:** How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?
        **Answer:** Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs.
        **Metadata:** Company: Apple | SEC-Filing: 10-K
        **Factoid:** Apple continues to invest in renewable energy projects, carbon offset initiatives, and innovative low-carbon product designs to meet its sustainability goals.


        ### Example Input/Output (JSON):
        Response: {
            "support": "yes",
            "reasoning": "Higher upfront investments in sustainability - Directly affects cost structure by increasing capital expenditures on renewable energy and carbon offset projects"
        }
        
        ### Input for your task:
        """
        for factoid in factoid_list:
            instruction_prompt = instruction_prompt + f"\n**Question:** {qna_dict['query']}\n**Answer:** {qna_dict['answer']}\n**Metadata:** {metadata}\n**Factoid:**{factoid}"
            print('prompt length: ', len(instruction_prompt))
            #summary, messages = self.__execute_LLM_task_chain(instruction_prompt, max_new_tokens=5000, temperature=0, top_p = 0.9, return_prompt_messages = True, prev_messages = qna_messages)
            print('Generated Response for groundings:\n', summary)
        return None


    def __generate_query_answer_pair(self, fact_doc_text, metadata, no_of_trials = 10):

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
        Metadata: Company name: AAPL | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Factoids: {["Apple committed to carbon neutrality across its supply chain by 2030.",
            "Apple sources renewable energy for its global operations.",
            "Apple integrates recycled materials into product design.",
            "The company works with suppliers to reduce emissions.",
            "Upfront costs have increased due to sustainability investments."
        ]}

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
        - Keep the generated answer concise and under 200 words.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response.
        - **Do not** put your chain of thought or reasoning steps in the response. Return **just the answer** in your final response.
        - **Do not copy example from the prompt** in your response.
        - Don't think for more than 3000 tokens.

        ### Input format:
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]
        Query: <query text>

        ### Output format (JSON):
        Answer: {
            "answer": <answer to the question, generated from fact(s) in the given text document>
        }

        ### Example Input
        Metadata: Company name: AAPL | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
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
        qstn_prompt_tokens = [self.__get_prompt_token(qstn_instruction_prompt) for _ in range(no_of_trials)]
        qoutputs = self.__execute_LLM_tasks(qstn_prompt_tokens, max_new_tokens=3000, temperature=0.2, top_p=0.9)

        #summary, qmessages = self.__execute_LLM_task_chain(qstn_instruction_prompt, max_new_tokens=3000, temperature=0.2, top_p = 0.9, return_prompt_messages = True)
        #print('Generated Response:\n', summary)
        qna_pairs = []
        query_strs = []
        for j, o in enumerate(qoutputs):
            qsummary = o.outputs[0].text.strip()
            print(f'generated {j}th response for question: ', qsummary)
            if "Query:" in qsummary:
                qi = qsummary.index("Query:")
                if "\n" in qsummary[(qi+6):]:
                    nli = qsummary.index("\n")
                    query_str = qsummary[(qi+6):nli].strip()
                    print('Generated query: ', query_str)
                    if self.__is_valid_sentence(query_str):
                        query_strs.append(query_str)
        answer_instruction_prompts = [answer_instruction_prompt + f"\nMetadata: {metadata}\nFactoids: {fact_doc_text}\nQuery: {q}" for q in query_strs]
        answer_prompt_tokens = [self.__get_prompt_token(prompt_text=prompt) for prompt in answer_instruction_prompts]
        aoutputs = self.__execute_LLM_tasks(answer_prompt_tokens, max_new_tokens=4096, temperature=0.2, top_p = 0.9)
        for query_str,o in zip(query_strs, aoutputs):
            ans_summary = o.outputs[0].text.strip()
            print('Generated answer response:\n', ans_summary)
            answer_json = self.__extract_json_text_by_key(ans_summary, "answer")
            if answer_json != None:
                qna_pair = {
                    'query': query_str,
                    'answer': answer_json["answer"]
                }
                qna_pairs.append(qna_pair)
        ans_strs = [qob['answer'] for qob in qna_pairs]
        cleaned_answers = self.__cleanup_sentence(ans_strs)
        cleaned_qna_pairs = []
        for ci,cleaned_ans in enumerate(cleaned_answers):
            if cleaned_ans != "":
                cleaned_qna_pairs.append({
                    'query': qna_pairs[ci]['query'],
                    'answer': cleaned_ans
                })
        return cleaned_qna_pairs


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
            metadata = f'Company: {self.company_abbr} | SEC Filing: 10-K | Related topic: {SEED_METADATA_TOPICS[self.topic_index]}'
            #for idx in random_indices:
                #factoid_subarr.append(all_factoids[idx])
            all_resp = []
            print('\nStarting query generation for batch of factoids\n')
            for i in range(0, len(filtered_factoids), MAX_FACTOIDS_TO_SAMPLE):
                factoid_subarr = filtered_factoids[i:i+MAX_FACTOIDS_TO_SAMPLE]
                if len(factoid_subarr) >= 15:
                    factoid_str = "[" + ",\n".join(f"{item['factoid']}" for item in factoid_subarr) + "]"
                    print(f'\nRunning {no_of_trials} qna generation for factoids batch {i}')
                    qna_pairs = self.__generate_query_answer_pair(factoid_str, metadata, no_of_trials)
                    print('No of valid qna pairs: ', len(qna_pairs))
                    for qna_pair in qna_pairs:
                        query_dict = self.__generate_grounding_reasoning_pair(factoid_str, metadata, qna_pair, no_of_trials)
                        if query_dict != None and "reasonings" in query_dict and "groundings" in query_dict:
                            query_dict = query_dict | qna_pair
                            all_resp.append(query_dict)
                        
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
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    query_gen = QueryGenerator(filename = args.filename, model_index = args.model_index, topic_index = args.topic_index)
    query_gen.generate_query(no_of_trials = args.no_of_trials)
