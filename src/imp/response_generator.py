from typing import List
from src.interface.base_response_generator import BaseResponseGenerator
from src.utils.invoke_ai import invoke_ai

SYSTEM_PROMPT = """
You are answering factual questions about Alice's Adventures in Wonderland based on provided context.

Instructions:
- Give SHORT, DIRECT answers (1-5 words when possible)
- Extract the answer even if it's mentioned briefly or indirectly in the context
- Look for synonyms, descriptions, or related mentions of the answer
- Be confident - if the context contains relevant information, extract it
- ONLY respond "I cannot find the answer in the context" if the context truly has NO relevant information
- Do not add extra explanations unless necessary

Examples:
Q: "What does Alice drink?" → A: "A bottle labeled DRINK ME"
Q: "Who smokes a hookah?" → A: "The Caterpillar"
Q: "What animal can disappear?" → A: "The Cheshire Cat" (even if just mentions "grinning cat" or "cat that vanishes")
Q: "What time at tea party?" → A: "Six o'clock" (even if says "tea time" or "always six")
"""


class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using OpenAI's chat completion."""
        # Combine context into a single string
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)