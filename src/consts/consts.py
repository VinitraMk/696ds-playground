
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
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "meta-llama/llama-3.3-70b-versatile"
]


HF_CACHE_DIR = '/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.huggingface-cache'

NO_OF_TRIALS = 3

FILENAMES = [
    '10-K_AMD_20231230',
    '10-K_NVDA_20240128',
    '10-K_F_20231231',
    '10-K_GM_20231231',
    '10-K_INTC_20231230',
    '10-K_TSLA_20231231'
]

IGNORE_ENTITIES = ['TABLE OF CONTENTS', 'SEC', '10-K FILINGS', 'SEC 10-K FILINGS', 'SEC 10-K', 'SEC (Securities and Exchange Commission)', 'NOTES', 'ITEM 1A', 'PART IV', 'ITEM 15', 'ITEM 601(B)(32)(II)', 'ITEM 15', 'ITEM', 'CONSOLIDATED FINANCIAL STATEMENTS', 'FORM 10-K', 'ANNUAL REPORT', 'SECURITIES AND EXCHANGE COMMISSION']
MAX_GROUNDINGS_TO_SAMPLE = 50
MIN_GROUNDINGS_NEEDED_FOR_GENERATION = 10

QUERY_TYPE_MAP = {
    'tmp': 'temporal_analysis',
    'evtint': 'event_interaction_analysis',
    'etint': 'entity_interaction_analysis',
    'num': 'numerical_analysis',
    'summ': 'summarization',
    'none': 'none'
}

QUERY_INDEX = {
    0: 'temporal_analysis',
    1: 'event_interaction_analysis',
    2: 'entity_interaction_analysis',
    3: 'numerical_analysis',
    4: 'summarization'
}