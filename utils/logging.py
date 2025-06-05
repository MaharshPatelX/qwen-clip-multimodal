import logging
import sys
from pathlib import Path
import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a basic one."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create basic logger if none exists
        logger = setup_logger(name)
    
    return logger


def setup_training_logger(experiment_name: str, output_dir: str) -> logging.Logger:
    """
    Set up logger specifically for training with timestamped file.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for logs
        
    Returns:
        Configured training logger
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/training_{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=f"training.{experiment_name}",
        log_file=log_file,
        level=logging.INFO
    )


def setup_evaluation_logger(output_dir: str) -> logging.Logger:
    """Set up logger for evaluation."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/evaluation_{timestamp}.log"
    
    return setup_logger(
        name="evaluation",
        log_file=log_file,
        level=logging.INFO
    )


class TrainingProgressLogger:
    """Helper class for logging training progress."""
    
    def __init__(self, logger: logging.Logger, log_interval: int = 100):
        self.logger = logger
        self.log_interval = log_interval
        self.step_count = 0
        
    def log_step(self, loss: float, lr: float, **metrics):
        """Log training step metrics."""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            log_msg = f"Step {self.step_count}: loss={loss:.4f}, lr={lr:.2e}"
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_msg += f", {key}={value:.4f}"
                else:
                    log_msg += f", {key}={value}"
            
            self.logger.info(log_msg)
    
    def log_epoch(self, epoch: int, avg_loss: float, **metrics):
        """Log epoch summary."""
        log_msg = f"Epoch {epoch}: avg_loss={avg_loss:.4f}"
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_msg += f", {key}={value:.4f}"
            else:
                log_msg += f", {key}={value}"
        
        self.logger.info(log_msg)


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logger("test", "test.log")
    logger.info("Test log message")
    
    progress_logger = TrainingProgressLogger(logger, log_interval=5)
    
    for i in range(10):
        progress_logger.log_step(loss=0.5 - i*0.05, lr=1e-4, accuracy=0.7 + i*0.02)
    
    progress_logger.log_epoch(epoch=1, avg_loss=0.25, accuracy=0.85)
    
    print("Logging utilities working correctly!")