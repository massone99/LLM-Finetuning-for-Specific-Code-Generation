import json
from pathlib import Path
import pytest
from src.helpers import save_evaluation_results  # updated import path

def test_save_evaluation_results(tmp_path):
    test_results = [{"key": "value"}]
    output_file = tmp_path / "evaluation_results.json"
    
    # Invoke the method with a temporary file path
    save_evaluation_results(test_results, output_file)
    
    # Verify the file was created and contains the correct data
    data = json.loads(output_file.read_text())
    assert data == test_results
