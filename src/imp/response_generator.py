from typing import List
from src.interface.base_response_generator import BaseResponseGenerator
from src.utils.invoke_ai import invoke_ai

SYSTEM_PROMPT = """
You are an intelligent assistant answering questions about "Alice's Adventures in Wonderland".

Instructions:
- You will be provided with a question and a set of context passages.
- Your goal is to answer the question ACCURATELY based ONLY on the provided context.
- Match your answer style to the question type:
  * For simple "Who", "What", "Where", "When" questions: Give SHORT, DIRECT answers (1-10 words).
  * For "Why", "How", and "Explain" questions: Provide the reasoning or cause-and-effect (1-3 sentences).
- Be precise and factual. Do not add information that is not in the context.
- If the answer is NOT in the context, say "I cannot find the answer in the context".

Examples:
Q: "Who is smoking a hookah?"
A: "The Caterpillar."

Q: "What does Alice drink?"
A: "A bottle labeled 'DRINK ME'."

Q: "Why does Alice cry?"
A: "She cries because she shrank after drinking from the bottle and could no longer reach the key to the garden, leaving her trapped."
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