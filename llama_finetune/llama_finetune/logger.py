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
        """
        Write text to the log file.

        Args:
            text (str): The text to write.
            heading (int, optional): The heading level. Defaults to 0.
        """
        heading_prefix = '=' * heading
        self.log.write(f'\n{heading_prefix} {text}\n')

    def write_and_print(self, text, heading: int = 0):
        self.write(text, heading)
        print(text, end='')

import os

# Print the current working directory
print("Current working directory:", os.getcwd())

file_logger = FileLogger('../res/report/results.adoc')