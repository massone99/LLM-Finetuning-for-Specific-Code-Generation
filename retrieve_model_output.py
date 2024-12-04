import json
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk

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

def save_to_file(directory, idx, prompt, code):
    """
    Save the prompt and code to separate files within the specified directory.
    
    Parameters:
        directory (str): The directory where files will be saved.
        idx (int): The index of the current result (for file naming).
        prompt (str): The extracted prompt.
        code (str): The extracted Scala code.
    """
    os.makedirs(directory, exist_ok=True)
    
    # Define file paths
    prompt_file = os.path.join(directory, f'prompt_{idx}.txt')
    code_file = os.path.join(directory, f'scala_code_{idx}.scala')
    
    # Save prompt
    with open(prompt_file, 'w', encoding='utf-8') as pf:
        pf.write(prompt)
    
    # Save code
    with open(code_file, 'w', encoding='utf-8') as cf:
        cf.write(code)

def process_detailed_results(detailed_results, output_dir, status_display):
    """
    Process each detailed result to extract and save prompts and generated code.
    
    Parameters:
        detailed_results (list): List of detailed result dictionaries.
        output_dir (str): Directory to save the extracted prompts and codes.
        status_display (tk.Text): Text widget to display status messages.
    """
    total = len(detailed_results)
    if total == 0:
        status_display.insert(tk.END, "No detailed results found in the JSON data.\n")
        return

    for idx, result in enumerate(detailed_results, 1):
        generated_text = result.get('generated', '')
        
        prompt, code = extract_prompt_and_code(generated_text)
        
        if not prompt:
            prompt = 'Prompt not found.'
        if not code:
            code = 'Generated code not found.'
        
        # Display the extracted information
        status_display.insert(tk.END, f"\nResult {idx} of {total}:\n")
        status_display.insert(tk.END, "Prompt:\n")
        status_display.insert(tk.END, f"{prompt}\n\n")
        status_display.insert(tk.END, "Generated Scala Code:\n")
        status_display.insert(tk.END, f"{code}\n")
        status_display.insert(tk.END, "-" * 80 + "\n")
        
        # Save to files
        try:
            save_to_file(output_dir, idx, prompt, code)
            status_display.insert(tk.END, f"Saved Result {idx} to files.\n")
        except Exception as e:
            status_display.insert(tk.END, f"Error saving Result {idx}: {e}\n")
        
        # Scroll to the end
        status_display.see(tk.END)
        status_display.update_idletasks()

    status_display.insert(tk.END, "\nProcessing Complete.\n")
    status_display.see(tk.END)

def select_json_file():
    """
    Open a file dialog to select a JSON file.
    """
    file_path = filedialog.askopenfilename(
        title="Select JSON File",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    return file_path

def select_output_directory():
    """
    Open a directory dialog to select an output directory.
    """
    directory = filedialog.askdirectory(
        title="Select Output Directory"
    )
    return directory

def start_processing(json_path, output_dir, status_display, process_button):
    """
    Initiate the processing of the selected JSON file.
    
    Parameters:
        json_path (str): Path to the JSON file.
        output_dir (str): Directory to save the extracted data.
        status_display (tk.Text): Text widget to display status messages.
        process_button (tk.Button): Button widget to start processing.
    """
    if not json_path:
        messagebox.showerror("Error", "Please select a JSON file to process.")
        return

    if not output_dir:
        # Default output directory
        output_dir = os.path.join(os.getcwd(), 'extracted_outputs')

    # Disable the process button to prevent multiple clicks
    process_button.config(state=tk.DISABLED)
    status_display.insert(tk.END, f"Loading JSON file: {json_path}\n")
    status_display.update_idletasks()

    # Load JSON data
    data, error = load_json(json_path)
    if error:
        status_display.insert(tk.END, f"{error}\n")
        messagebox.showerror("Error", error)
        process_button.config(state=tk.NORMAL)
        return

    # Get detailed results
    detailed_results = data.get('detailed_results', [])
    if not detailed_results:
        msg = "No detailed results found in the JSON data."
        status_display.insert(tk.END, f"{msg}\n")
        messagebox.showwarning("Warning", msg)
        process_button.config(state=tk.NORMAL)
        return

    # Start processing
    status_display.insert(tk.END, "Starting extraction process...\n")
    status_display.see(tk.END)
    status_display.update_idletasks()

    process_detailed_results(detailed_results, output_dir, status_display)

    # Re-enable the process button
    process_button.config(state=tk.NORMAL)

def create_gui():
    """
    Create the GUI window using Tkinter.
    """
    root = tk.Tk()
    root.title("LLM Output Extractor")
    root.geometry("800x600")
    root.resizable(False, False)

    # Configure grid layout
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)
    root.columnconfigure(2, weight=1)

    # JSON File Selection
    json_label = ttk.Label(root, text="JSON File:")
    json_label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

    json_path_var = tk.StringVar()
    json_entry = ttk.Entry(root, textvariable=json_path_var, width=60)
    json_entry.grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)

    json_browse_button = ttk.Button(root, text="Browse", command=lambda: browse_json(json_path_var))
    json_browse_button.grid(column=2, row=0, padx=10, pady=10)

    # Output Directory Selection
    output_label = ttk.Label(root, text="Output Directory:")
    output_label.grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)

    output_dir_var = tk.StringVar()
    output_entry = ttk.Entry(root, textvariable=output_dir_var, width=60)
    output_entry.grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)

    output_browse_button = ttk.Button(root, text="Browse", command=lambda: browse_output(output_dir_var))
    output_browse_button.grid(column=2, row=1, padx=10, pady=10)

    # Start Processing Button
    process_button = ttk.Button(root, text="Start Processing", command=lambda: start_processing(
        json_path_var.get(),
        output_dir_var.get(),
        status_display,
        process_button
    ))
    process_button.grid(column=1, row=2, padx=10, pady=20)

    # Status Display Area
    status_label = ttk.Label(root, text="Status:")
    status_label.grid(column=0, row=3, padx=10, pady=10, sticky=tk.NW)

    status_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=25, state='normal')
    status_display.grid(column=0, row=4, columnspan=3, padx=10, pady=10)

    return root

def browse_json(json_path_var):
    """
    Handle the browsing of the JSON file.
    """
    file_path = select_json_file()
    if file_path:
        json_path_var.set(file_path)

def browse_output(output_dir_var):
    """
    Handle the browsing of the output directory.
    """
    directory = select_output_directory()
    if directory:
        output_dir_var.set(directory)

def main():
    """
    Main function to create and run the GUI.
    """
    root = create_gui()
    root.mainloop()

if __name__ == "__main__":
    main()
