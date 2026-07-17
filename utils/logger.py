import logging
import os
import sys
import json
from typing import Dict, Any

def setup_logger(name: str = "DualPathVLM", log_file: str = "train.log"):
    """
    Sets up a logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # 1. Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # 2. File handler
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 3. Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # 4. Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

class ExperimentLogger:
    """
    Simple JSON-based experiment logger for research tracking.
    """
    def __init__(self, log_dir: str = "./outputs/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, "metrics.json")
        self.metrics = []
        
        # Load existing metrics if present
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                self.metrics = json.load(f)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Logs metrics with current step.
        """
        metrics["step"] = step
        self.metrics.append(metrics)
        
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
