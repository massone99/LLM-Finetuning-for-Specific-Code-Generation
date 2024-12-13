import os
import json
import subprocess
import shutil
import uuid
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import queue
import logging
from logging.handlers import QueueHandler, QueueListener

# Configuration parameters
SBT_VERSION = "1.8.2"  
AKKA_VERSION = "2.6.20"
SCALA_VERSION = "2.13.10"

LOG_FILE = "build.log"

class AkkaSbtBuilderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Akka sbt Project Builder")
        master.geometry("800x600")

        # Initialize variables
        self.dataset_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="akka_sbt_projects")
        self.build_projects = tk.BooleanVar()
        self.run_projects = tk.BooleanVar()

        # Create a queue for thread-safe logging
        self.log_queue = queue.Queue()

        # Set up logging
        self.setup_logging()

        # Create and place GUI components
        self.create_widgets()

        # Start polling the log queue
        self.master.after(100, self.process_log_queue)

    def setup_logging(self):
        # Create logger
        self.logger = logging.getLogger("AkkaSbtBuilder")
        self.logger.setLevel(logging.DEBUG)

        # Create formatters
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # GUI handler via Queue
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(queue_handler)

        # We don't need a QueueListener since the GUI thread reads from the queue directly.

    def create_widgets(self):
        # Dataset Selection
        dataset_frame = tk.Frame(self.master)
        dataset_frame.pack(pady=10, padx=10, fill=tk.X)

        dataset_label = tk.Label(dataset_frame, text="Dataset JSON File:")
        dataset_label.pack(side=tk.LEFT)

        dataset_entry = tk.Entry(dataset_frame, textvariable=self.dataset_path, width=60)
        dataset_entry.pack(side=tk.LEFT, padx=5)

        browse_dataset_btn = tk.Button(dataset_frame, text="Browse", command=self.browse_dataset)
        browse_dataset_btn.pack(side=tk.LEFT)

        # Output Directory Selection
        output_frame = tk.Frame(self.master)
        output_frame.pack(pady=10, padx=10, fill=tk.X)

        output_label = tk.Label(output_frame, text="Output Directory:")
        output_label.pack(side=tk.LEFT)

        output_entry = tk.Entry(output_frame, textvariable=self.output_dir, width=60)
        output_entry.pack(side=tk.LEFT, padx=5)

        browse_output_btn = tk.Button(output_frame, text="Browse", command=self.browse_output_dir)
        browse_output_btn.pack(side=tk.LEFT)

        # Options: Build and Run
        options_frame = tk.Frame(self.master)
        options_frame.pack(pady=10, padx=10, fill=tk.X)

        build_cb = tk.Checkbutton(options_frame, text="Build Projects (sbt compile)", variable=self.build_projects)
        build_cb.pack(anchor=tk.W)

        run_cb = tk.Checkbutton(options_frame, text="Run Projects (sbt run)", variable=self.run_projects)
        run_cb.pack(anchor=tk.W)

        # Start Button
        start_btn = tk.Button(self.master, text="Start Processing", command=self.start_processing, bg="green", fg="white")
        start_btn.pack(pady=20)

        # Status Display
        status_label = tk.Label(self.master, text="Status:")
        status_label.pack(anchor=tk.W, padx=10)

        self.status_text = scrolledtext.ScrolledText(self.master, height=20, width=90, state='disabled', wrap=tk.WORD)
        self.status_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

    def browse_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.dataset_path.set(file_path)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def process_log_queue(self):
        # Process log messages from the queue and update the GUI
        while not self.log_queue.empty():
            record = self.log_queue.get()
            self.display_log_record(record)
        self.master.after(100, self.process_log_queue)

    def display_log_record(self, record):
        """
        Display log record in GUI. Color-coded and with emojis based on level.
        """
        self.status_text.configure(state='normal')

        if record.levelno == logging.INFO:
            prefix = "ℹ️ "
            tag = "info"
            color = "blue"
        elif record.levelno == logging.WARNING:
            prefix = "⚠️ "
            tag = "warning"
            color = "orange"
        elif record.levelno == logging.ERROR:
            prefix = "❌ "
            tag = "error"
            color = "red"
        elif record.levelno == logging.CRITICAL:
            prefix = "🔥 "
            tag = "critical"
            color = "red"
        else:
            # DEBUG or other levels
            prefix = "🐛 "
            tag = "debug"
            color = "grey"

        self.status_text.tag_config(tag, foreground=color)
        formatted_message = f"{prefix}{record.getMessage()}\n"
        self.status_text.insert(tk.END, formatted_message, tag)
        self.status_text.see(tk.END)
        self.status_text.configure(state='disabled')

        # Also print to console
        if record.levelno == logging.INFO:
            print(f"ℹ️ {record.getMessage()}")
        elif record.levelno == logging.WARNING:
            print(f"⚠️ {record.getMessage()}")
        elif record.levelno == logging.ERROR:
            print(f"❌ {record.getMessage()}")
        elif record.levelno == logging.CRITICAL:
            print(f"🔥 {record.getMessage()}")
        else:
            print(f"🐛 {record.getMessage()}")

    def load_dataset(self, json_file_path):
        if not os.path.exists(json_file_path):
            self.logger.error(f"The file '{json_file_path}' does not exist.")
            messagebox.showerror("File Not Found", f"The file '{json_file_path}' does not exist.")
            return None

        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
            self.logger.info(f"Successfully loaded dataset from '{json_file_path}'.")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON file: {e}")
            messagebox.showerror("JSON Decode Error", f"Error decoding JSON file:\n{e}")
            return None

    def create_sbt_project(self, project_name, scala_code, parent_dir, scala_version, akka_version):
        unique_id = uuid.uuid4().hex[:8]
        project_dir = os.path.join(parent_dir, f"{project_name}_{unique_id}")

        try:
            os.makedirs(project_dir, exist_ok=False)
        except FileExistsError:
            self.logger.warning(f"Directory {project_dir} already exists. Skipping.")
            return None

        # Directory structure
        src_main_scala = os.path.join(project_dir, "src", "main", "scala")
        os.makedirs(src_main_scala, exist_ok=True)

        # Write build.sbt
        build_sbt_content = f"""
name := "{project_name}"

version := "0.1"

scalaVersion := "{scala_version}"

libraryDependencies += "com.typesafe.akka" %% "akka-actor-typed" % "{akka_version}"
libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "{akka_version}"
libraryDependencies += "com.typesafe.akka" %% "akka-slf4j" % "{akka_version}"

// For logging
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.11"

// Enable auto-reloading for run
// addSbtPlugin("com.typesafe.sbteclipse" % "sbteclipse-plugin" % "5.2.4")
        """

        with open(os.path.join(project_dir, "build.sbt"), "w") as f:
            f.write(build_sbt_content.strip())

        # Write build.properties
        build_properties_content = f"sbt.version={SBT_VERSION}\n"
        project_subdir = os.path.join(project_dir, "project")
        os.makedirs(project_subdir, exist_ok=True)

        with open(os.path.join(project_subdir, "build.properties"), "w") as f:
            f.write(build_properties_content)

        # Determine main object
        main_object = None
        for line in scala_code.splitlines():
            line = line.strip()
            if line.startswith("object "):
                parts = line.split()
                if len(parts) >= 2:
                    main_object = (
                        parts[1].split("{")[0].split("(")[0]
                    )  # Handle cases like 'object Main extends App {'
                    break
        if not main_object:
            main_object = "Main"

        scala_file_name = f"{main_object}.scala"
        with open(os.path.join(src_main_scala, scala_file_name), "w") as f:
            f.write(scala_code.strip())

        self.logger.info(f"Created sbt project: {project_dir}")
        return project_dir

    def build_sbt_project(self, project_dir):
        self.logger.info(f"Building project: {project_dir}")
        try:
            result = subprocess.run(
                ["sbt", "compile"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False
            )
            if result.returncode == 0:
                self.logger.info(f"Successfully compiled project: {project_dir}")
                return True
            else:
                self.logger.error(f"Failed to compile project: {project_dir}")
                self.logger.error(f"Error Output:\n{result.stderr}")
                return False
        except FileNotFoundError:
            self.logger.error("sbt is not installed or not found in PATH.")
            messagebox.showerror("sbt Not Found", "sbt is not installed or not found in PATH.")
            return False

    def run_sbt_project(self, project_dir):
        self.logger.info(f"Running project: {project_dir}")
        try:
            result = subprocess.run(
                ["sbt", "run"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False
            )
            if result.returncode == 0:
                self.logger.info(f"Successfully ran project: {project_dir}")
                self.logger.info(f"Output:\n{result.stdout}")
                return True
            else:
                self.logger.error(f"Failed to run project: {project_dir}")
                self.logger.error(f"Error Output:\n{result.stderr}")
                return False
        except FileNotFoundError:
            self.logger.error("sbt is not installed or not found in PATH.")
            messagebox.showerror("sbt Not Found", "sbt is not installed or not found in PATH.")
            return False

    def start_processing(self):
        self.status_text.configure(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.configure(state='disabled')

        dataset_file = self.dataset_path.get()
        output_directory = self.output_dir.get()
        build_flag = self.build_projects.get()
        run_flag = self.run_projects.get()

        if not dataset_file:
            messagebox.showwarning("Input Required", "Please select a dataset JSON file.")
            return
        if not output_directory:
            messagebox.showwarning("Input Required", "Please select an output directory.")
            return

        # Start processing in a separate thread to keep GUI responsive
        processing_thread = threading.Thread(
            target=self.process_projects,
            args=(dataset_file, output_directory, build_flag, run_flag),
            daemon=True  # Daemonize thread to exit with the main program
        )
        processing_thread.start()

    def process_projects(self, dataset_file, output_directory, build_flag, run_flag):
        dataset = self.load_dataset(dataset_file)
        if dataset is None:
            return

        os.makedirs(output_directory, exist_ok=True)
        self.logger.info(f"Output directory set to: {output_directory}")

        # Iterate over each conversation
        for idx, conversation in enumerate(dataset):
            assistant_messages = [
                msg["value"]
                for msg in conversation.get("conversations", [])
                if msg.get("from") == "assistant"
            ]

            # Iterate over each assistant message
            for msg_idx, scala_code in enumerate(assistant_messages):
                project_name = f"akka_project_{idx+1}_{msg_idx+1}"
                project_dir = self.create_sbt_project(
                    project_name=project_name,
                    scala_code=scala_code,
                    parent_dir=output_directory,
                    scala_version=SCALA_VERSION,
                    akka_version=AKKA_VERSION,
                )

                if project_dir:
                    # Build the project if flag is set
                    if build_flag:
                        build_success = self.build_sbt_project(project_dir)
                        if not build_success:
                            self.logger.error("Build failed. Stopping further processing.")
                            messagebox.showerror("Build Failed", f"Build failed for project: {project_dir}\nProcessing has been stopped.")
                            return  # Stop processing further projects

                    # Run the project if flag is set
                    if run_flag and build_flag:  # Only run if build was successful
                        run_success = self.run_sbt_project(project_dir)
                        if not run_success:
                            self.logger.error("Run failed. Stopping further processing.")
                            messagebox.showerror("Run Failed", f"Run failed for project: {project_dir}\nProcessing has been stopped.")
                            return  # Stop processing further projects

        self.logger.info("All projects processed successfully.")
        messagebox.showinfo("Processing Complete", "All projects have been processed successfully.")

def main():
    root = tk.Tk()
    gui = AkkaSbtBuilderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
