import argparse
import json
import os
import re
import sys
import torch
from time import time
from vllm import LLM, SamplingParams

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

SEED_METADATA_TOPICS = [
    "Leadership & Governance", "ESG & Sustainability", "Risk Factors",
    "Financial Strategy", "Business Growth & Strategy", "Technology & Innovation"
]

HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'
os.environ['HF_HOME'] = HF_CACHE_DIR


class BottomUpQueryGenerator:
    def __init__(self, filename, model_index=0, chunk_length=500,
                 max_tokens=200, min_factoids=5, max_factoids=20):
        self.filename = filename
        self.page_data = {}
        self.model_index = model_index
        self.chunk_length = chunk_length
        self.max_tokens = max_tokens
        self.min_factoids = min_factoids
        self.max_factoids = max_factoids
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Device enabled: {self.device}')

        # Load the model using vLLM
        model_name = MODELS[self.model_index]
        self.llm = LLM(model_name=model_name, tensor_parallel_size=torch.cuda.device_count())

    def __extract_numbered_bullets(self, text, add_newline=False):
        bullet_pattern = r"^\s*\d+[\.\)-]\s+"
        lines = text.split("\n")
        numbered_bullets = [re.sub(bullet_pattern, "", line).strip() + ("\n" if add_newline else "")
                            for line in lines if re.match(bullet_pattern, line)]
        return numbered_bullets

    def __filter_relevant_chunks(self, chunks, metadata_topic="Financial Strategy"):
        instruction_prompt = f"""
        ### Task:
        Given a chunk of text and a topic, determine if the text is relevant to the topic.
        Respond with "yes" or "no".

        ### Input:
        - Topic: {metadata_topic}
        """

        relevant_chunks = []
        for chunk in chunks:
            prompt = instruction_prompt + f"\n- Text: {chunk}"
            response = self.__generate_text(prompt, max_new_tokens=5)
            print(f'Chunk relevance response: {response.strip()}')
            if response.strip().lower() == "yes":
                relevant_chunks.append(chunk)
        return relevant_chunks

    def __generate_text(self, prompt, max_new_tokens):
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            stop=["###"]
        )
        outputs = self.llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text if outputs else ""

    def __generate_factoids(self, chunk_data, category="Financial Strategy"):
        instruction = (
            f"A factoid is a concise, factual statement that captures a key point from the text.\n"
            f"You are an AI assistant summarizing factoids related to {category} from SEC 10-K filings.\n"
            "Present the generated factoids as a numbered list.\n\n"
            "### TEXT:\n"
            f"{chunk_data}\n\n"
            "### FACTOIDS:\n1."
        )

        summary = self.__generate_text(instruction, max_new_tokens=300)
        print('Generated Factoids:\n', summary)
        return self.__extract_numbered_bullets(summary, add_newline=True)

    def __generate_metadata(self, chunk_data):
        instruction = (
            f"### INSTRUCTION:\nGenerate metadata topics covered in the following text.\n"
            "List each metadata topic as a numbered bullet.\n\n"
            "### TEXT:\n"
            f"{chunk_data}\n\n"
            "### METADATA TOPICS:\n1."
        )

        summary = self.__generate_text(instruction, max_new_tokens=self.max_tokens)
        return self.__extract_numbered_bullets(summary)

    def __generate_queries(self, document_text):
        instruction = (
            f"### INSTRUCTION:\nGenerate 5 multi-factoid queries answerable using the given text.\n"
            "For each query, provide:\n"
            "- The query itself.\n"
            "- An answer derived from factoids in the text.\n"
            "- A list of grounding factoids (by index).\n"
            "- Reasoning explaining the answer.\n\n"
            "### TEXT:\n"
            f"{document_text}\n\n"
            "### QUERIES:\n1."
        )

        summary = self.__generate_text(instruction, max_new_tokens=1024)
        print('Generated Queries:\n', summary)

    def run(self, instruction_type=INSTRUCTION_TYPES['FACTOIDS']):
        st = time()

        # Load chunked data
        chunk_fp = f'data/chunked_data/10-K_AMD_20231230_chunked.json'
        with open(chunk_fp, 'r') as fp:
            self.all_chunks = json.load(fp)['chunks']

        # Filter relevant chunks
        self.chunks = self.__filter_relevant_chunks(self.all_chunks)

        all_resp = []
        if instruction_type == INSTRUCTION_TYPES['FACTOIDS']:
            for ci, text in enumerate(self.chunks):
                print(f'\nProcessing chunk {ci}...')
                generated_factoids = self.__generate_factoids(text, "Financial Strategy")
                all_resp.extend(generated_factoids)

            txt_file = f'data/factoids/factoids-{self.filename}.txt'
            all_resp.insert(0, '### FACTOIDS:\n')
            with open(txt_file, 'w') as fp:
                fp.writelines(all_resp)

        elif instruction_type == INSTRUCTION_TYPES['QUERIES']:
            factoid_file = f'data/factoids/factoids-{self.filename}.txt'
            with open(factoid_file, 'r') as fp:
                file_contents = fp.read()
            self.__generate_queries(file_contents)

        print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_index', type=int, default=1, required=False)
    parser.add_argument('--min_factoids', type=int, default=5, required=False)
    parser.add_argument('--max_factoids', type=int, default=10, required=False)
    parser.add_argument('--max_chunk_length', type=int, default=5000, required=False)
    parser.add_argument('--max_tokens', type=int, default=4096, required=False)
    parser.add_argument('--instruction_type', type=str, default=INSTRUCTION_TYPES['FACTOIDS'], required=False)

    args = parser.parse_args()

    filename = '000000248824000012-amd-20231230'
    bu_query_gen = BottomUpQueryGenerator(
        filename=filename, model_index=args.model_index,
        chunk_length=args.max_chunk_length, max_tokens=args.max_tokens,
        min_factoids=args.min_factoids, max_factoids=args.max_factoids
    )
    bu_query_gen.run(instruction_type=args.instruction_type)

