from google import genai
import os
import multiprocessing
import torch
import argparse
from time import time
import sys
import json
import re

#API_KEY1 = "AIzaSyDjjGKGfr5-wOH7aCG7p8U6kvvR0yVp_14"
API_KEY = "AIzaSyALRwfCf7GEnX7XlLRfEPZ6VfCwQdEaC3M"

COMPANY_DICT = {
    'INTC': 'Intel Corp.',
    'AMD': 'AMD Inc.',
    'NVDA': 'Nvidia Corp.',
    'TSLA': 'Tesla Inc.',
    'F': 'Ford Motor Company',
    'GM': 'General Motors'
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
MAX_FACTOIDS_TO_SAMPLE = 25
MIN_FACTOIDS_NEEDED_FOR_GENERATION = 15


class GeminiQueryGenerator:

    def __init__(self, filename, model_index = 0, topic_index = 0):
        self.filename = filename
        self.topic_index = topic_index
        self.company_abbr = COMPANY_DICT[filename.split('_')[1]]
        self.client = genai.Client(api_key=API_KEY)


    def __execute_LLM_task(self, instruction_prompt):
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{instruction_prompt}",
        )
        print(response.text)
        return response.text
    
    def __extract_query_set(self, text):
        """
        Extract a JSON object with the keys 'query', 'answer', 'groundings', and 'reasonings'
        from raw LLM output text.
        """
        # Regex pattern for the full object, including arrays
        pattern = r'\{\s*"query"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*,\s*"groundings"\s*:\s*\[(.*?)\]\s*,\s*"reasonings"\s*:\s*\[(.*?)\]\s*\}'

        match = re.search(pattern, text, re.DOTALL)
        if match:
            query = match.group(1).strip()
            answer = match.group(2).strip()

            try:
                # Safely load the list-like portions
                groundings = json.loads("[" + match.group(3).strip() + "]")
                reasonings = json.loads("[" + match.group(4).strip() + "]")
            except json.JSONDecodeError as e:
                print("JSON parsing error in groundings or reasonings:", e)
                return None

            return {
                "query": query,
                "answer": answer,
                "groundings": groundings,
                "reasonings": reasonings
            }

        return None

    def __generate_query_set(self, factoid_doc_text, metadata):
        query_set_instruction_prompt = """
        Analyze the provided set of factoids and the metadata and generate **only one structured response** as described below.

        ### Desired response structure:  
        {
            "query": <a complex question based on one or multiple factoids>,
            "answer": <answer to the question>,
            "groundings": <list of factoids supporting the answer>,
            "reasonings": <for each point in the grounding, explain concisely why/how it supports the answer to the question>
        }
        

        ### Generation Rules
        - Phrase your response as concisely as possible.
        - Keep the query under 100 words and the answer under 200 words.
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.
        - Use the example as reference of question complexity, answer style and how "groundings" and "reasoning" support the answer.

        ### Input format:
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]

        ### Output format:
        {
            "query": <a complex question based on one or multiple factoids>,
            "answer": <answer to the question>,
            "groundings": <list of factoids supporting the answer>,
            "reasonings": <for each point in the grounding, explain concisely why/how it supports the answer to the question>
        }

        ### Example Input:
        Metadata: Company: Apple | SEC-Filing: 10-K
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
        
        ### Example Output:
        {
            "query": "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?",
            "answer": "Apple’s environmental initiatives have a significant impact on its cost structure, supplier relationships, and long-term profitability. Upfront costs have risen due to investments in renewable energy, low-carbon manufacturing, and sustainable materials. However, the shift to clean energy and more energy-efficient processes may reduce operational costs over time. Additionally, integrating recycled and sustainable materials into product development increases design complexity and processing requirements, thereby raising costs. On the supply side, Apple mandates carbon reduction compliance from its suppliers, leading to higher compliance costs for those partners. Some suppliers may struggle to meet ESG standards, which could cause supply chain disruptions and increase component expenses. Nonetheless, Apple is strengthening its relationships with sustainable suppliers, gaining long-term cost advantages and fostering innovation. From a profitability perspective, Apple benefits from an enhanced brand reputation, which appeals to environmentally conscious consumers and supports premium pricing. Early adoption of ESG practices also mitigates future risks associated with carbon taxes and tighter environmental regulations. Although short-term profitability may be affected by increased expenses, the long-term financial gains are expected to outweigh these initial investments.",
            "groundings": [“Apple continues to invest in renewable energy projects, carbon offset initiatives, and innovative low-carbon product designs to meet its sustainability goals.”,
                “The company has deployed 100% renewable energy for corporate operations and is working with suppliers to expand clean energy use across the entire supply chain.”,
                “Apple’s commitment to carbon neutrality includes increasing the use of recycled materials such as recycled aluminum, rare earth elements, and plastics in its products.”,
                “Apple requires its suppliers to transition to clean energy and reduce emissions, with financial support and guidance provided where necessary.”,
                “Suppliers that fail to meet Apple’s sustainability requirements may face discontinuation of partnerships.”,
                “Apple has partnered with key suppliers to develop low-carbon manufacturing processes and co-invest in renewable energy projects.”,
                “Apple’s commitment to carbon neutrality reinforces its brand reputation for environmental responsibility, which continues to drive customer loyalty.”,
                “By proactively aligning with global carbon reduction goals, Apple reduces exposure to future regulatory penalties and carbon taxes.”,
                “Apple anticipates that ongoing sustainability investments may impact operating expenses in the near term.”]
            ]
            "reasonings": ["Higher upfront investments in sustainability - Directly affects cost structure by increasing capital expenditures on renewable energy and carbon offset projects.",
                "Operational cost savings through energy efficiency - Supports long-term cost reduction, counterbalancing high initial expenses.",
                "Higher product development and material costs - Increases short-term costs, but necessary for achieving carbon neutrality.",
                "Suppliers must comply with strict carbon reduction policies - Introduces higher compliance costs for suppliers, impacting supply chain relationships.",
                "Risk of supply chain disruptions due to ESG non-compliance - Potential supplier instability, leading to increased sourcing costs.",
                "Strengthening supplier partnerships for low-carbon innovation - Mitigates supply risks by co-investing in sustainable solutions, creating cost efficiencies over time.",
                "Enhanced brand reputation and consumer loyalty - Drives demand for Apple products, helping offset increased costs through premium pricing.",
                "Competitive and regulatory advantage through early ESG adoption - Protects against future carbon taxes and compliance risks, reducing long-term financial risk exposure.",
                "Short-term cost pressure but long-term profitability upside - Confirms initial profitability impact, but strategic benefits could lead to higher long-term margins."
            ]
        }

        ### Input for your task:
        """

        query_set_instruction_prompt = query_set_instruction_prompt + f"\nMetadata: {metadata}\nFactoids: {factoid_doc_text}"
        query_set_response = self.__execute_LLM_task(query_set_instruction_prompt)
        return self.__extract_query_set(query_set_response)

    def generate_query(self, no_of_qstns = 5):
        st = time()

        log_fp = f'logs/bu-gemini-query-logs.txt'
        log_file = open(log_fp, 'w')
        old_stdout = sys.stdout
        sys.stdout = log_file

        # Filter relevant chunks

        all_resp = []
        #no_of_trials = 10

        chunk_store_fp = f'data/chunked_data/global_chunk_store/qwq/{self.filename}_chunk_store.json'
        
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
            no_of_attempts = 0
            while ((len(all_resp) < no_of_qstns) and (no_of_attempts < 5)):
                for i in range(0, len(filtered_factoids), MAX_FACTOIDS_TO_SAMPLE):
                    factoid_subarr = filtered_factoids[i:i+MAX_FACTOIDS_TO_SAMPLE]
                    if len(factoid_subarr) < MIN_FACTOIDS_NEEDED_FOR_GENERATION:
                        factoid_subarr = filtered_factoids
                    factoid_str = "[" + ",\n".join(f"{item['factoid']}" for item in factoid_subarr) + "]"
                    query_set = self.__generate_query_set(factoid_str, metadata)
                    if query_set != None:
                        all_resp.append(query_set)
                    if (len(all_resp) == no_of_qstns):
                        break
                no_of_attempts += 1

            print('No of valid whole set: ', len(all_resp))
            query_json_path = f'./data/queries/gemini/{self.filename}_gen_queries.json'
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

            with open(f'./data/queries/gemini/{self.filename}_gen_queries.json', 'w') as fp:
                json.dump(queries, fp) 
        else:
            print('Chunk store not found!')
            SystemExit()

        print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
        sys.stdout = old_stdout
        log_file.close()


if __name__ == "__main__":
    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    #torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 5, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)

    args = parser.parse_args()

    query_gen = GeminiQueryGenerator(filename = args.filename, model_index = args.model_index, topic_index = args.topic_index)
    query_gen.generate_query(args.no_of_qstns)
