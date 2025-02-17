import json
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QFileDialog, QLabel, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
import sys

ROOT_PATH = Path("/home/lorix/Documenti/uni/tesi/finetuning-llama3.2-scala-for-dsl")
DEFAULT_DATASET_DIR = ROOT_PATH / "llama_finetune/res/data"

def load_json_file(file_path: str) -> list:
    """Load JSON file and return its content."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: list, file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

class TrimDatasetGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.input_file = None
        self.dataset = None
        
    def initUI(self):
        self.setWindowTitle('Dataset Trimmer')
        self.setGeometry(300, 300, 500, 250)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Input file selection
        self.input_label = QLabel('No input file selected')
        select_input_btn = QPushButton('Select Input Dataset')
        select_input_btn.clicked.connect(self.select_input_file)
        
        # Dataset size controls
        self.current_size_label = QLabel('Current dataset size: -')
        size_label = QLabel('Target dataset size:')
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(1)
        self.size_spinbox.setMaximum(10000)
        
        # Trim button
        self.trim_btn = QPushButton('Trim Dataset')
        self.trim_btn.clicked.connect(self.trim_dataset)
        self.trim_btn.setEnabled(False)
        
        # Add widgets to layout
        layout.addWidget(select_input_btn)
        layout.addWidget(self.input_label)
        layout.addWidget(self.current_size_label)
        layout.addWidget(size_label)
        layout.addWidget(self.size_spinbox)
        layout.addWidget(self.trim_btn)
        
        # Add some spacing
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

    def select_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            str(DEFAULT_DATASET_DIR),  # Changed from Path.home()
            "JSON Files (*.json)"
        )
        
        if file_name:
            try:
                self.input_file = file_name
                self.dataset = load_json_file(file_name)
                current_size = len(self.dataset)
                
                self.input_label.setText(f'Selected: {file_name}')
                self.current_size_label.setText(f'Current dataset size: {current_size}')
                
                self.size_spinbox.setMaximum(current_size)
                self.size_spinbox.setValue(current_size)
                
                self.trim_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error loading file: {str(e)}')
                self.reset_state()

    def trim_dataset(self):
        if not self.dataset or not self.input_file:
            return
            
        target_size = self.size_spinbox.value()
        trimmed_dataset = self.dataset[:target_size]
        
        # Get output filename with default in the same directory as input
        input_path = Path(self.input_file)
        default_name = str(input_path.parent / f"{input_path.stem}_size{target_size}.json")
        
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trimmed Dataset",
            default_name,
            "JSON Files (*.json)",
            options=QFileDialog.DontUseNativeDialog  # To ensure we can set the initial directory
        )
        
        if output_file:
            try:
                save_json_file(trimmed_dataset, output_file)
                QMessageBox.information(
                    self,
                    'Success',
                    f'Trimmed dataset saved to: {output_file}\nNew size: {len(trimmed_dataset)}'
                )
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error saving file: {str(e)}')

    def reset_state(self):
        self.input_file = None
        self.dataset = None
        self.input_label.setText('No input file selected')
        self.current_size_label.setText('Current dataset size: -')
        self.size_spinbox.setValue(1)
        self.trim_btn.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    gui = TrimDatasetGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
