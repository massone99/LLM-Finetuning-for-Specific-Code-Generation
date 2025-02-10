import requests
import json
import os
from typing import Tuple
from logger import file_logger


class BuildCheckerClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def process_dataset_inline_content(
        self, data_content, build: bool = False, run: bool = True
    ) -> Tuple[int, int]:
        """
        Process a dataset directly from memory via the inline endpoint.
        """
        payload = {
            "data": data_content,
            "build": build,
            "run": run,
            "use_hashes": False,
        }
        file_logger.write_and_print("Sending request to build checker API inline content\n")

        response = requests.post(f"{self.base_url}/process-dataset-inline", json=payload)
        if response.status_code != 200:
            file_logger.write_and_print(
                f"API Error ({response.status_code}): {response.text}"
            )
            return 0, 0

        result = response.json()
        return result["successful_runs"], result["total_snippets"]


