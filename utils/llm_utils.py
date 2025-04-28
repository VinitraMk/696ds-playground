from transformers import AutoTokenizer
from vllm import SamplingParams

def get_prompt_token(prompt_text, system_prompt_text, model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}")
    tokenizer.lang = "en"

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