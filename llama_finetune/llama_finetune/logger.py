import sys

class FileLogger:
    """
    Logger class that writes to a file in
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        try:
            self.log = open(filename, 'w')
        except Exception as e:
            print("Current working directory:", os.getcwd())
            raise e
    
    def write(self, text, heading: int = 0):
        heading_prefix = '=' * heading
        self.log.write(f'\n{heading_prefix} {text}\n')

    def write_and_print(self, text, heading: int = 0):
        self.write(text, heading)
        print(text, end='')

import os

# Get the absolute path to the log file
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../res/report/results.adoc')

print(f"Log file path: {log_file_path}")

# Print the log file path using the logger
file_logger = FileLogger(log_file_path)