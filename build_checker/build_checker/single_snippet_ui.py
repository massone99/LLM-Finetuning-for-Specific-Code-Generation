
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from .api import BuildCheckerAPI

class ProcessSingleSnippetWindow(QMainWindow):
    def __init__(self, api: BuildCheckerAPI):
        super().__init__()
        self.api = api
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Test Snippet")
        self.setGeometry(200, 200, 800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.code_input = QTextEdit()
        self.code_input.setPlaceholderText("Paste your Scala code here...")
        
        self.test_button = QPushButton("Test Snippet")
        self.test_button.clicked.connect(self.test_snippet)
        
        self.result_label = QLabel("")
        
        layout.addWidget(self.code_input)
        layout.addWidget(self.test_button)
        layout.addWidget(self.result_label)

    def test_snippet(self):
        code = self.code_input.toPlainText()
        success, output = self.api.test_single_snippet(code)
        
        if success:
            self.result_label.setText("Success!")
        else:
            self.result_label.setText(f"Error: {output}") # Print full output to stdout