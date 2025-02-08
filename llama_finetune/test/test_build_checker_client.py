import json
import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from llama_finetune.evaluate import BuildCheckerClient

@pytest.mark.integration
def test_process_dataset_real_api(tmp_path, start_server):
    test_code = """
    object HelloWorld {
        def main(args: Array[String]): Unit = {
            println("Hello, World!")
        }
    }
    """
    test_file = tmp_path / "test_dataset.json"
    test_data = [{
        "conversations": [
            {"from": "human", "value": "Write a Hello World program in Scala"},
            {"from": "assistant", "value": test_code}
        ]
    }]

    with open(test_file, "w") as f:
        json.dump(test_data, f)

    with open(test_file, "r") as f:
        data_content = json.load(f)

    client = BuildCheckerClient(base_url="http://localhost:8000")
    successful_runs, total_snippets = client.process_dataset_inline_content(data_content)

    print(f"\nTest file path: {test_file}")
    print(f"Successful runs: {successful_runs}")
    print(f"Total snippets: {total_snippets}")
    assert total_snippets > 0, "Expected at least one snippet to be processed"