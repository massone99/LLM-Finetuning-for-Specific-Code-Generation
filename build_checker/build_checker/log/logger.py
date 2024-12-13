import logging

# Define color codes
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m',# Magenta
        'RESET': '\033[0m'    # Reset
    }

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Configure the logging system
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("exec.log", mode="w")

# Create formatters
color_formatter = ColorFormatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Set formatters
console_handler.setFormatter(color_formatter)
file_handler.setFormatter(file_formatter)

# Configure logger
logger = logging.getLogger("GeneralLogger")

logger.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def process_items(items):
    logger.info(f"Starting to process {len(items)} items.")

    for idx, item in enumerate(items, start=1):
        try:
            if item % 2 == 0:
                logger.debug(f"Processing item {idx}: {item}")
            else:
                logger.warning(f"Item {idx}: {item} is odd, might require extra attention.")

            if item == 5:
                raise ValueError(f"Item {idx}: {item} caused an error!")

        except ValueError as e:
            logger.error(f"Error encountered while processing item {idx}: {e}")
            break

    logger.info("Processing completed.")

if __name__ == "__main__":
    items = [2, 3, 4, 5, 6]
    process_items(items)