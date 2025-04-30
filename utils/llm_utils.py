from transformers import AutoTokenizer
from vllm import SamplingParams
import torch

def get_tokenizer(model_name):
    if "Llama-3.3-70B" in model_name:
        model_path = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)
        tokenizer.lang = "en"
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}")
        tokenizer.lang = "en"
    return tokenizer

def get_prompt_token(prompt_text, system_prompt_text, tokenizer):
    messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": prompt_text}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

def execute_LLM_tasks(llm_model, prompts, max_new_tokens, temperature = 0.3, top_p = 0.9):
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        stop=["###EOF###"]
    )
    #model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
    outputs = llm_model.generate(prompts, sampling_params)
    #response_text = tokenizer.decode(
        #outputs[0].outputs[0].token_ids,
        #skip_special_tokens=True,
        #clean_up_tokenization_spaces=True  # Ensures proper spacing
    #)
    #return outputs[0].outputs[0].text if outputs else ""
    return outputs

def execute_gemini_LLM_task(llm_model, instruction_prompt):
    response = llm_model.models.generate_content(
        model = "gemini-2.0-flash",
        contents = f"{instruction_prompt}"
    )
    return response.text

def execute_llama_LLM_task(llm_model, prompts, tokenizer, max_new_tokens, temperature = 0.3):
    outputs = llm_model.generate(
        input_ids = prompts["input_ids"],
        attention_mask = prompts["attention_mask"],
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        do_sample = True
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    return responses