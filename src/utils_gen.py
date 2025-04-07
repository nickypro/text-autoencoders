# %%
# Assume openai>=1.0.0
from openai import OpenAI
from dotenv import load_dotenv
import requests
import asyncio
import os
import random
import time
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import itertools
from utils_parallel import exponential_backoff, process_in_parallel


# Load token from env file
load_dotenv("./env")

API_TOKEN = os.environ.get("OPENROUTER_API_KEY")
MODEL_REPO = "meta-llama/llama-3.2-3b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"

openai = OpenAI(
    api_key=API_TOKEN,
    base_url=BASE_URL,
)

@exponential_backoff
def get_llama_completion(prompt: str) -> str:
    chat_completion = openai.chat.completions.create(
        model=MODEL_REPO,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content

def get_prompts_parallel(prompts: List[str], max_workers: int = 10) -> List[str]:
    def get_prompt_for_text(text: str) -> str:
        try:
            return get_llama_completion(text)
        except Exception as e:
            print(f"Error getting completion: {e}")
            return ""

    return process_in_parallel(items=prompts, process_func=get_prompt_for_text, max_workers=max_workers)


# %%
