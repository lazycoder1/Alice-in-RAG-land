from src.interface.base_evaluator import BaseEvaluator, EvaluationResult
from src.utils.invoke_ai import invoke_ai
from src.utils.extract_xml import extract_xml_tag

SYSTEM_PROMPT = """
You are evaluating answers about Alice's Adventures in Wonderland.

Instructions:
- Check if the RESPONSE contains the same CORE INFORMATION as the expected answer
- Be FLEXIBLE with wording - focus on factual correctness, not exact phrasing
- Accept equivalent answers (e.g., "White Rabbit" = "a White Rabbit" = "the White Rabbit")
- Accept longer answers if they include the key fact
- If the response says "I cannot find the answer", mark it as FALSE

Examples of CORRECT matches:
- Expected: "Lewis Carroll" | Response: "The author is Lewis Carroll" → TRUE
- Expected: "DRINK ME" | Response: "A bottle labeled 'DRINK ME'" → TRUE  
- Expected: "The Caterpillar" | Response: "Caterpillar" → TRUE
- Expected: "Hookah" | Response: "A hookah" → TRUE

Examples of INCORRECT:
- Expected: "Red" | Response: "I cannot find the answer" → FALSE
- Expected: "Flamingos" | Response: "Hedgehogs" → FALSE

Return your reasoning in <reasoning>...</reasoning> tags.
Then return the result in <result>...</result> tags — either 'true' or 'false'.
"""


class Evaluator(BaseEvaluator):
    def evaluate(self, query: str, response: str, expected_answer: str) -> EvaluationResult:
        """Evaluates the correctness of a response to a question"""
        try:
            user_prompt = f"""
                <question>\n{query}\n</question>
                <response>\n{response}\n</response>
                <expected_answer>\n{expected_answer}\n</expected_answer>
                """
            response_content = invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_prompt)

            reasoning = extract_xml_tag(response_content, "reasoning")
            result = extract_xml_tag(response_content, "result")
            print(f"✅ Evaluated response for question: {query}")
            print(f"✅ Reasoning: {reasoning}")
            print(f"✅ Result: {result}")

            if result is not None:
                is_correct = result.lower() == "true"
            else:
                is_correct = False
                reasoning = f"No result found: ({response_content})"

            return EvaluationResult(
                question=query,
                response=response,
                expected_answer=expected_answer,
                is_correct=is_correct,
                reasoning=reasoning
            )
        except Exception as e:
            print(f"❌ Error evaluating response for question: {query}")
            print(f"❌ Error: {e}")
            return EvaluationResult(
                question=query,
                response=response,
                expected_answer=expected_answer,
                is_correct=False,
                reasoning=f"Error evaluating response: ({e})"
            )
