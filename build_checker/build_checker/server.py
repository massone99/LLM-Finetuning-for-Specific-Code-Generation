from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from api import BuildCheckerAPI
import os

app = FastAPI(title="Build Checker API")
api = BuildCheckerAPI()

class CodeSnippet(BaseModel):
    code: str
    build: bool = True
    run: bool = True

class Dataset(BaseModel):
    file_path: str
    build: bool = True
    run: bool = True
    use_hashes: bool = False

class SnippetResponse(BaseModel):
    success: bool
    message: str

class ProcessResponse(BaseModel):
    successful_runs: int
    total_snippets: int
    completed: bool
    message: str

class InlineDataset(BaseModel):
    data: List
    build: bool = True
    run: bool = True
    use_hashes: bool = False

@app.post("/test-snippet", response_model=SnippetResponse)
async def test_snippet(snippet: CodeSnippet):
    """Test a single Scala code snippet"""
    success, message = api.test_single_snippet(
        snippet.code,
        build=snippet.build,
        run=snippet.run
    )
    return SnippetResponse(success=success, message=message)


@app.post("/process-dataset-inline", response_model=ProcessResponse)
async def process_dataset_inline(dataset: InlineDataset):
    """Process a dataset of code snippets passed directly as JSON"""
    try:
        data = dataset.data
        if not data:
            raise HTTPException(
                status_code=400,
                detail="No data provided in the dataset"
            )

        # Extract code snippets from conversations
        snippets = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'conversations' in item:
                    for conv in item['conversations']:
                        if isinstance(conv, dict) and 'value' in conv:
                            snippets.append(conv['value'])

        # Process all snippets
        successful_runs, total_snippets = api.process_snippets(
            data,
            dataset.build,
            dataset.run,
            dataset.use_hashes
        )

        print(f"Processed {successful_runs}/{len(snippets)} snippets successfully")

        return ProcessResponse(
            successful_runs=successful_runs,
            total_snippets=len(snippets),
            completed=True,
            message=f"Processed {successful_runs}/{len(snippets)} snippets successfully"
        )
    except Exception as e:
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

def start_server(host="localhost", port=8000):
    """Start the FastAPI server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
