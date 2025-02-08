import sys
import os
import subprocess
import pytest
import time

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add the directory containing the logger module to sys.path
logger_path = os.path.join(project_root, 'llama_finetune')
sys.path.append(logger_path)

@pytest.fixture(scope="session")
def start_server():
    # Start the server
    server_process = subprocess.Popen([
        "poetry", "run", "python", "../build_checker/build_checker/server.py"
    ])
    time.sleep(5)  # Wait for the server to start

    yield

    # Teardown: Stop the server
    server_process.terminate()
    server_process.wait()