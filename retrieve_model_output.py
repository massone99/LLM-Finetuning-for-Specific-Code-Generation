import json
import re
import os

def load_json(file_path):
    """
    Load JSON data from the specified file path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data, None
    except FileNotFoundError:
        return None, f"Error: The file {file_path} was not found."
    except json.JSONDecodeError as e:
        return None, f"Error decoding JSON: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def extract_prompt_and_code(generated_text):
    """
    Extract the prompt and the generated code from the given generated_text.
    
    Parameters:
        generated_text (str): The text containing the prompt and generated code.
        
    Returns:
        tuple: A tuple containing the prompt and the generated code.
    """
    # Define the patterns
    prompt_pattern = r'user\s*\n\n(.*?)\s*assistant\s*\n\n'
    code_pattern = r'assistant\s*\n\n(.*)'
    
    # Extract prompt
    prompt_match = re.search(prompt_pattern, generated_text, re.DOTALL | re.IGNORECASE)
    prompt = prompt_match.group(1).strip() if prompt_match else None

    # Extract code
    code_match = re.search(code_pattern, generated_text, re.DOTALL | re.IGNORECASE)
    code = code_match.group(1).strip() if code_match else None

    return prompt, code

def save_to_file(directory, idx, prompt, code, file_extension='.txt'):
    """
    Save the prompt and code to separate files within the specified directory.
    
    Parameters:
        directory (str): The directory where files will be saved.
        idx (int): The index of the current result (for file naming).
        prompt (str): The extracted prompt.
        code (str): The extracted code.
        file_extension (str): The extension for the code file (default: .txt).
    """
    os.makedirs(directory, exist_ok=True)
    
    # Define file paths
    prompt_file = os.path.join(directory, f'prompt_{idx}.txt')
    code_file = os.path.join(directory, f'code_{idx}{file_extension}')
    
    # Save prompt
    with open(prompt_file, 'w', encoding='utf-8') as pf:
        pf.write(prompt)
    
    # Save code
    with open(code_file, 'w', encoding='utf-8') as cf:
        cf.write(code)
    
    return prompt_file, code_file

def process_evaluation_results(json_path, output_dir, file_extension='.scala'):
    """
    Process evaluation results from a JSON file and save extracted prompts and code.
    
    Parameters:
        json_path (str): Path to the JSON file containing evaluation results.
        output_dir (str): Directory to save the extracted prompts and code.
        file_extension (str): The extension for the code files (default: .scala).
    
    Returns:
        tuple: (success: bool, message: str, saved_files: list)
    """
    # Load JSON data
    data, error = load_json(json_path)
    if error:
        return False, error, []

    # Get detailed results
    detailed_results = data.get('detailed_results', [])
    if not detailed_results:
        return False, "No detailed results found in the JSON data.", []

    saved_files = []
    for idx, result in enumerate(detailed_results, 1):
        generated_text = result.get('generated', '')
        prompt, code = extract_prompt_and_code(generated_text)
        
        if not prompt:
            prompt = 'Prompt not found.'
        if not code:
            code = 'Generated code not found.'
        
        # Save to files
        try:
            prompt_file, code_file = save_to_file(output_dir, idx, prompt, code, file_extension)
            saved_files.extend([prompt_file, code_file])
        except Exception as e:
            return False, f"Error saving files for result {idx}: {str(e)}", saved_files

    success_msg = f"Successfully processed {len(detailed_results)} results and saved to {output_dir}"
    return True, success_msg, saved_files

if __name__ == "__main__":
    # Example usage
    json_path = "path/to/evaluation_results.json"
    output_dir = "path/to/output"
    success, message, files = process_evaluation_results(json_path, output_dir)
    print(message)
    if success:
        print("Saved files:", files)
