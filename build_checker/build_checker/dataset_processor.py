from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, QFileDialog
from api import BuildCheckerAPI
from log.logger import logger

class DatasetProcessorGUI(QMainWindow):
    def __init__(self, api: BuildCheckerAPI):
        super().__init__()
        self.api = api
        self.selected_file = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.select_button = QPushButton("Select Dataset")
        self.select_button.clicked.connect(self.select_file)

        self.build_checkbox = QCheckBox("Build Project")
        self.run_checkbox = QCheckBox("Run Project")
        self.hash_checkbox = QCheckBox("Use Hashes")

        self.process_button = QPushButton("Process Dataset")
        self.process_button.clicked.connect(self.process_dataset)

        self.status_label = QLabel("")

        layout.addWidget(self.select_button)
        layout.addWidget(self.build_checkbox)
        layout.addWidget(self.run_checkbox)
        layout.addWidget(self.hash_checkbox)
        layout.addWidget(self.process_button)
        layout.addWidget(self.status_label)

        self.setGeometry(300, 300, 400, 200)
        self.setWindowTitle('Dataset Processor')

    def select_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "JSON Files (*.json)")
        if filename:
            self.selected_file = filename
            self.status_label.setText(f"Selected: {filename}")

    def process_dataset(self):
        logger.info("Processing dataset started")
        if not self.selected_file:
            logger.warning("No file selected")
            self.status_label.setText("Please select a dataset first")
            return

        self.status_label.setText("Processing...")
        dataset = self.api.load_json_dataset(self.selected_file)
        
        print(f"\033[92mLoaded data: {dataset}\033[0m")
        
        if dataset:
            success, total = self.api.process_snippets(
                dataset,
                self.build_checkbox.isChecked(),
                self.run_checkbox.isChecked(),
                self.hash_checkbox.isChecked()
            )
            self.status_label.setText(f"Processed {success}/{total} snippets successfully")
        else:
            self.status_label.setText("Error loading dataset")
