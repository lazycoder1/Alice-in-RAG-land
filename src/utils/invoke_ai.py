import os
from openai import OpenAI

def invoke_ai(system_message: str, user_message: str) -> str:
    openai_client = OpenAI()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    )
    return response.choices[0].message.content
