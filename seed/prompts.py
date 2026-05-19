# seed/prompts.py
from typing import List, Union

class PromptManager:
    @staticmethod
    def get_system_prompt() -> str:
        """
        System Prompt extracted from the official baseline tests
        """
        return "You are a helpful assistant for visual question answering. Use the image and the text to answer multiple-choice questions."

    @staticmethod
    def get_m3cot_prompt(question: str, choices: Union[List[str], str] = None, context: str = "") -> str:
        """
        Multiple-choice prompt template fully aligned with the baseline tests
        """
        options_str = ""
        valid_letters = "A/B/C/D" # Default fallback

        # Strictly align with the option-construction logic of the reference code
        if isinstance(choices, list):
            opts_dict = {chr(65 + i): str(v) for i, v in enumerate(choices)}
            options_str = "\n".join([f"{k}. {v}" for k, v in opts_dict.items()])
            valid_letters = "/".join(opts_dict.keys())
        elif isinstance(choices, str):
            options_str = choices
            
        # Handle the case where context may be present (include it if present, leave empty otherwise)
        context_str = f"Context: {context}\n\n" if context and context.strip() else ""

        prompt = (
            "You are a helpful visual question answering assistant. "
            "You must answer multiple-choice questions about an image.\n\n"
            f"{context_str}"
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
            "Please think step by step, then output your final answer in the format: "
            f"Answer: <{valid_letters}>.\n"
            "Let's think step by step."
        )
        return prompt 