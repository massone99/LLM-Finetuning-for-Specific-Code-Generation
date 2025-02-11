import json
import os

class ConfigManager:
    def __init__(self, config_path="res/config/gui_config.json"):
        self.config_path = config_path
        self.default_config = {
            "build_enabled": False,
            "run_enabled": True, 
            "hash_enabled": False,
            "append_enabled": False,
            "input_dataset": "",
            "main_dataset": ""
        }
        
    def load_config(self):
        """Load configuration from file or return defaults"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
        return self.default_config.copy()

    def save_config(self, config):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
