import re
import os
import json
from typing import List, Dict
import pandas as pd

class DataProcessor:
    @staticmethod
    def extract_conversations(convo_list: List[Dict]) -> pd.Series:
        """Extract human and assistant messages from conversation list."""
        try:
            human_msg = convo_list[0]["value"]
            assistant_msg = convo_list[1]["value"]
            return pd.Series({"prompt": human_msg, "reference": assistant_msg})
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error extracting conversations: {e}")
            return pd.Series({"prompt": None, "reference": None})

    @staticmethod
    def convert_pairs_to_json(folder_path: str) -> str:
        """Convert prompt-code pairs to JSON format."""
        file_pattern = re.compile(r"^(prompt|code)_(\d+)\.txt$")
        prompts = {}
        codes = {}

        for file_name in os.listdir(folder_path):
            match = file_pattern.match(file_name)
            if match:
                file_type, idx = match.group(1), match.group(2)
                with open(
                    os.path.join(folder_path, file_name), "r", encoding="utf-8"
                ) as f:
                    content = f.read().strip()

                if file_type == "prompt":
                    prompts[idx] = content
                else:
                    codes[idx] = content

        result = []
        all_indices = sorted(set(prompts.keys()).union(codes.keys()), key=int)

        for idx in all_indices:
            prompt_text = prompts.get(idx, "")
            code_text = codes.get(idx, "")
            if prompt_text or code_text:
                result.append(
                    {
                        "conversations": [
                            {"from": "human", "value": prompt_text},
                            {"from": "assistant", "value": code_text},
                        ]
                    }
                )

        return json.dumps(result, indent=2)