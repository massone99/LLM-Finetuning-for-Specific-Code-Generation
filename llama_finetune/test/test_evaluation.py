import os
import tempfile
import shutil
import pytest


@pytest.fixture
def setup_test_dir():
    test_dir = tempfile.mkdtemp()
    code_1 = os.path.join(test_dir, 'code_1.txt')
    code_2 = os.path.join(test_dir, 'code_2.txt')
    prompt_1 = os.path.join(test_dir, 'prompt_1.txt')
    prompt_2 = os.path.join(test_dir, 'prompt_2.txt')
    
    hello_world_code = 'object HelloWorld {\n  def main(args: Array[String]): Unit = {\n    println("Hello, world!")\n  }\n}\n'
    hello_world_prompt = 'Write a Scala program that prints "Hello, world!" to the console.'
    
    with open(code_1, 'w') as f:
        f.write(hello_world_code)
    
    with open(code_2, 'w') as f:
        f.write(hello_world_code)
    
    with open(prompt_1, 'w') as f:
        f.write(hello_world_prompt)
    
    with open(prompt_2, 'w') as f:
        f.write(hello_world_prompt)
    
    yield test_dir
    
    shutil.rmtree(test_dir)


def test_convert_pairs_to_json(setup_test_dir):
    from llama_finetune.evaluate import convert_pairs_to_json
    result = convert_pairs_to_json(setup_test_dir)
    print(f"Result: {result}")
    assert result is not None


if __name__ == '__main__':
    pytest.main()