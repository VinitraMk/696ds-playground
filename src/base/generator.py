import torch
import os
import sys
import json
from time import time, sleep
from groq import AsyncGroq
from together import Together
from typing import List, Any

# custom imports
from utils.llm_utils import get_prompt_token, execute_LLM_tasks, get_tokenizer, execute_llama_task_api, execute_groq_task_api
from src.consts.company_consts import COMPANY_DICT
from src.consts.consts import MODELS

class Generator:

    def __init__(self, model_index:int = 11, prompt_batch_size:int = 1):
        self.model_index = model_index
        self.model_name = MODELS[self.model_index]
        
        self.prompt_batch_size = prompt_batch_size
        self.tokenizer = get_tokenizer(model_name = self.model_name)

        print('Model to be used: ', self.model_name)
        with open("./config.json", "r") as fp:
            cfg = json.load(fp)

        if self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
            torch.cuda.init()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            print('Invalid model name passed!')
            SystemExit()

    def get_output_from_llm(self, instruction_prompts: List[str], system_prompt: str, json_schema: Any = None, llm_config: Any = { 'temperature': 0.6, 'max_completion_tokens': 8192 }):
        summary = "",
        summary_stats = []
        if self.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct":
            prompt_tokens = self.tokenizer([get_prompt_token(instruction_prompt[0], system_prompt, self.tokenizer)], return_tensors = "pt", padding = True, truncation = True).to(self.device)
            outputs = execute_llama_LLM_task(self.llm, prompt_tokens, self.tokenizer, max_new_tokens=3000, temperature=0.6)
            summary = outputs[0]
            if "Input for your task" in summary:
                ti = summary.index("Input for your task")
                summary = summary[ti:]
            summary = [summary]
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":
            summary = execute_llama_task_api(self.llm, instruction_prompts[0], system_prompt)
            print('generated response: ', summary)
            summary = [summary]
        elif self.model_name == "meta-llama/llama-3.3-70b-versatile":
            summaries = execute_groq_task_api(
                llm_model = self.llm,
                response_format = json_schema,
                prompts = instruction_prompts,
                system_prompt = system_prompt,
                temperature = llm_config['temperature'],
                max_completion_tokens = llm_config['max_completion_tokens'])
            summary = [robj['response'] for robj in summaries]
            for robj in summaries:
                new_obj = {k: v for k,v in robj.items() if k != "responses" }
                summary_stats.append(new_obj)
            print('generated response: ', summary[0])
        else:
            print('Invalid model name passed!')
            SystemExit()

        return summary, summary_stats

    def set_filename(self, filecode: str = "NVDA"):
        self.filename = COMPANY_DICT[filecode]['filename']
        self.filecode = filecode
        self.company_name = COMPANY_DICT[filecode]['company_name']

    def get_script_status(self):
        script_status = {}
        with open('./script_status.json', 'r') as fp:
            script_status = json.load(fp)
        return script_status

    def update_script_status(self, script_status, k, v):
        script_status[k] = v
        with open('./script_status.json', 'w') as fp:
            json.dump(script_status, fp)
    
    def reset_script_status(self):
        with open('./script_status.json', 'w') as fp:
            json.dump({}, fp)


