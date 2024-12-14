import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import json
import os

class DatasetEditor:
    def __init__(self, master):
        self.master = master
        master.title("LLM Fine-Tuning Dataset Editor")

        self.dataset = []

        self.last_selected_conversation = None

        # Menu
        self.menu = tk.Menu(master)
        master.config(menu=self.menu)

        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.file_menu.add_command(label="New Dataset", command=self.new_dataset)
        self.file_menu.add_command(label="Load Dataset", command=self.load_dataset)
        self.file_menu.add_command(label="Save Dataset", command=self.save_dataset)
        self.file_menu.add_command(label="Save Dataset As", command=self.save_dataset_as)
        self.menu.add_cascade(label="File", menu=self.file_menu)

        # Listbox to display conversations
        self.conversation_listbox = tk.Listbox(master, width=60)
        self.conversation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.conversation_listbox.bind('<<ListboxSelect>>', self.display_conversation)

        # Scrollbar for the listbox
        self.scrollbar = tk.Scrollbar(master)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.conversation_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.conversation_listbox.yview)

        # Frame for entry fields
        self.entry_frame = tk.Frame(master)
        self.entry_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.prompt_label = tk.Label(self.entry_frame, text="Prompt:")
        self.prompt_label.pack()
        self.prompt_text = tk.Text(self.entry_frame, height=10)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        self.response_label = tk.Label(self.entry_frame, text="Response:")
        self.response_label.pack()
        self.response_text = tk.Text(self.entry_frame, height=10)
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # Buttons
        self.button_frame = tk.Frame(self.entry_frame)
        self.button_frame.pack(fill=tk.X)

        self.add_button = tk.Button(self.button_frame, text="Add", command=self.add_conversation)
        self.add_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.update_button = tk.Button(self.button_frame, text="Update", command=self.update_conversation)
        self.update_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.delete_button = tk.Button(self.button_frame, text="Delete", command=self.delete_conversation)
        self.delete_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.current_file = None

    def new_dataset(self):
        self.dataset = []
        self.current_file = None
        self.refresh_listbox()
        self.prompt_text.delete('1.0', tk.END)
        self.response_text.delete('1.0', tk.END)
        messagebox.showinfo("New Dataset", "Started a new dataset.")

    def load_dataset(self):
        file_path = filedialog.askopenfilename(defaultextension=".json",
                                               filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            self.current_file = file_path
            self.refresh_listbox()
            messagebox.showinfo("Load Dataset", f"Loaded dataset from {os.path.basename(file_path)}.")

    def save_dataset(self):
        if self.current_file:
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2)
            messagebox.showinfo("Save Dataset", f"Dataset saved to {os.path.basename(self.current_file)}.")
        else:
            self.save_dataset_as()

    def save_dataset_as(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2)
            self.current_file = file_path
            messagebox.showinfo("Save Dataset As", f"Dataset saved to {os.path.basename(file_path)}.")

    def add_conversation(self):
        prompt = self.prompt_text.get('1.0', tk.END).strip()
        response = self.response_text.get('1.0', tk.END).strip()

        if prompt and response:
            conversation = {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "assistant", "value": response}
                ]
            }
            self.dataset.append(conversation)
            self.refresh_listbox()
            self.prompt_text.delete('1.0', tk.END)
            self.response_text.delete('1.0', tk.END)
            messagebox.showinfo("Add Conversation", "Conversation added to the dataset.")
        else:
            messagebox.showwarning("Input Error", "Both prompt and response are required.")

    def update_conversation(self):
        selected = self.conversation_listbox.curselection() if self.conversation_listbox.curselection() else self.last_selected_conversation
        if selected:
            index = selected[0]
            prompt = self.prompt_text.get('1.0', tk.END).strip()
            response = self.response_text.get('1.0', tk.END).strip()

            if prompt and response:
                conversation = {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "assistant", "value": response}
                    ]
                }
                self.dataset[index] = conversation
                self.refresh_listbox()
                messagebox.showinfo("Update Conversation", "Conversation updated.")
            else:
                messagebox.showwarning("Input Error", "Both prompt and response are required.")
        else:
            messagebox.showwarning("Selection Error", "No conversation selected.")

    def delete_conversation(self):
        selected = self.conversation_listbox.curselection()
        if selected:
            index = selected[0]
            del self.dataset[index]
            self.refresh_listbox()
            self.prompt_text.delete('1.0', tk.END)
            self.response_text.delete('1.0', tk.END)
            messagebox.showinfo("Delete Conversation", "Conversation deleted.")
        else:
            messagebox.showwarning("Selection Error", "No conversation selected.")

    def refresh_listbox(self):
        self.conversation_listbox.delete(0, tk.END)
        for conv in self.dataset:
            prompt = conv["conversations"][0]["value"]
            display_text = prompt if len(prompt) <= 50 else prompt[:47] + '...'
            self.conversation_listbox.insert(tk.END, display_text)

    def display_conversation(self, event):
        selected = self.conversation_listbox.curselection()
        self.last_selected_conversation = selected
        if selected:
            index = selected[0]
            conversation = self.dataset[index]
            prompt = conversation["conversations"][0]["value"]
            response = conversation["conversations"][1]["value"]

            self.prompt_text.delete('1.0', tk.END)
            self.prompt_text.insert(tk.END, prompt)

            self.response_text.delete('1.0', tk.END)
            self.response_text.insert(tk.END, response)

def main():
    root = tk.Tk()
    DatasetEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
