import json
import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from llama_finetune.evaluate import BuildCheckerClient

'''
The tmp_path fixture is provided by pytest and is used to create temporary directories and files for testing purposes. 
It is automatically injected into the test function by pytest.
The directory and its contents are automatically cleaned up after the test function finishes.
'''

@pytest.mark.integration  # Mark as integration test so it can be run separately
def test_process_dataset_real_api(tmp_path):
    # Create a test file with some Scala code
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

    # Create client and process dataset
    client = BuildCheckerClient(base_url="http://localhost:8000")
    successful_runs, total_snippets = client.process_dataset(str(test_file))
    
    # Print debug information
    print(f"\nTest file path: {test_file}")
    print(f"Successful runs: {successful_runs}")
    print(f"Total snippets: {total_snippets}")
    
    # Assertions
    assert total_snippets > 0, "Expected at least one snippet to be processed"