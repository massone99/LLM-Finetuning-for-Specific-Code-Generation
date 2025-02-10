from src.main import PromptEvaluator
import pytest
from unittest.mock import patch


@pytest.fixture
def evaluator():
    return PromptEvaluator(model_name="test-model")


@pytest.fixture
def mock_response():
    return {"response": "test response"}


def test_prompt_evaluator_init(evaluator):
    assert evaluator.model_name == "test-model"
    assert evaluator.base_url == "http://localhost:11434/api"


# @patch('requests.post'): intercetta le chiamate HTTP POST
@patch("requests.post")
def test_generate_response_without_system_prompt(mock_post, evaluator, mock_response):
    mock_post.return_value.json.return_value = mock_response

    result = evaluator.generate_response("test prompt")

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={"model": "test-model", "prompt": "test prompt", "stream": False},
    )
    assert result == mock_response


@patch("requests.post")
def test_generate_response_with_system_prompt(mock_post, evaluator, mock_response):
    mock_post.return_value.json.return_value = mock_response

    result = evaluator.generate_response(
        "test prompt", system_prompt="test system prompt"
    )

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "test-model",
            "prompt": "test prompt",
            "system": "test system prompt",
            "stream": False,
        },
    )
    assert result == mock_response
