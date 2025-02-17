# Build Checker

This project provides tools for validating Scala code snippets with the following features:
- A FastAPI server exposing endpoints to test individual code snippets or process entire datasets.
- A PyQt5-based GUI to select datasets, configure build/run options, and view processing outcomes.
- An integrated API to compile, run, and log Scala code results, including error handling and failed snippet storage.
- Automatic append of working coding snippets to a dataset used for fine-tuning for fast testing of training data
- Support for dataset deduplication via processed hashes and configuration management.

# Usage
- Run the API server using the `server.py`.
- Use the GUI (or command line) for dataset processing using the `main.py`.
- Review logs and JSON reports for build/run failures.

# Project Structure
- **build_checker/**: Main folder containing API, server, GUI code, and configuration.
- **res/**: Contains configuration and output files.
- **res/src/main/scala/**: Destination for Scala code execution.

# Requirements
- Python 3.x, FastAPI, uvicorn, PyQt5, and sbt installed and in the PATH.

