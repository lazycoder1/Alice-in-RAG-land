from openai import OpenAI

def invoke_ai(system_message: str, user_message: str) -> str:
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    )
    return response.choices[0].message.content
