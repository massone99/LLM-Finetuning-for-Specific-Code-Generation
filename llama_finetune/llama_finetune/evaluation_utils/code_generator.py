from typing import Any, List
from unsloth import FastLanguageModel

class CodeGenerator:
    @staticmethod
    def generate_code(
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.5,
        min_p: float = 0.1,
    ) -> List[str]:
        """Generate code outputs for a list of prompts."""
        FastLanguageModel.for_inference(model)
        generated_codes = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
            )
            generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            generated_codes.append(generated_code)

        return generated_codes