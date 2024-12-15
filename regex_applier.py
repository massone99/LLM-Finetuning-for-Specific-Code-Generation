import re
import shutil  # Import shutil for file operations

# File paths
base_path = './dataset_builder/data/synthetic_data/'
input_file = base_path + 'dataset_llama.json'          # Replace with your input file path
output_file = base_path + 'dataset_llama_fixed.json'   # Replace with your desired output file path
backup_file = base_path + 'dataset_llama_BACKUP.json'  # Backup file path

# Create a backup of the input file
shutil.copyfile(input_file, backup_file)
print(f"Backup of input file created as {backup_file}")

# Read the input file
with open(input_file, 'r') as f:
    input_text = f.read()

# Regex pattern to match Props[...] but only if not followed by ()
pattern = r"Props\[[^\]]+\](?!\(\))"

# Function to replace using the pattern
def add_parentheses(match):
    content = match.group(0)  # The full match, e.g., Props[ToggleActor]
    inside_brackets = re.search(r'\[([^\]]+)\]', content).group(1)  # Extract content inside []
    return f"Props[{inside_brackets}]()"

# Perform the replacement
output_text = re.sub(pattern, add_parentheses, input_text)

# Write the modified text to the output file
with open(output_file, 'w') as f:
    f.write(output_text)

print(f"Modified file saved as {output_file}")