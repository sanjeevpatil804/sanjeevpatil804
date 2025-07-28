"""
Configuration file for StackGAN Text-to-Image Generation
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Text encoder
    vocab_size: int = 10000
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 1
    
    # Generator
    noise_dim: int = 100
    gf_dim: int = 128  # Generator feature dimension
    ef_dim: int = 128  # Embedding feature dimension
    
    # Discriminator
    df_dim: int = 128  # Discriminator feature dimension
    
    # Image sizes
    stage1_size: int = 64
    stage2_size: int = 256


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    batch_size: int = 16
    num_workers: int = 4
    image_size: int = 256
    max_text_length: int = 50
    
    # Optimization
    lr_g: float = 0.0002  # Generator learning rate
    lr_d: float = 0.0002  # Discriminator learning rate
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss weights
    lambda_kl: float = 2.0
    lambda_gp: float = 10.0  # Gradient penalty weight
    
    # Training
    num_epochs: int = 100
    save_interval: int = 10
    sample_interval: int = 5
    val_interval: int = 1
    
    # Logging
    log_interval: int = 100
    tensorboard: bool = True
    wandb: bool = False
    
    # Checkpointing
    save_best: bool = True
    save_latest: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    data_dir: str = "data/cub"
    train_split: str = "train"
    val_split: str = "val"
    
    # Augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    color_jitter: bool = True
    random_crop: bool = False
    
    # Text processing
    min_word_freq: int = 5
    max_vocab_size: int = 10000
    
    # Image processing
    normalize_mean: tuple = (0.5, 0.5, 0.5)
    normalize_std: tuple = (0.5, 0.5, 0.5)


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Model paths
    model_path: str = "checkpoints/stackgan_epoch_100.pth"
    vocab_path: str = "data/vocabulary.pkl"
    
    # Generation
    num_samples: int = 4
    temperature: float = 1.0
    noise_seed: Optional[int] = None
    
    # Output
    output_dir: str = "generated_images"
    save_format: str = "png"
    save_comparison: bool = True
    
    # Interactive
    interactive: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Directories
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"
    
    # TensorBoard
    tensorboard: bool = True
    tensorboard_port: int = 6006
    
    # Weights & Biases
    wandb: bool = False
    wandb_project: str = "stackgan-text-to-image"
    wandb_entity: Optional[str] = None
    
    # Console
    verbose: bool = True
    progress_bar: bool = True


@dataclass
class SystemConfig:
    """System configuration"""
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    cuda_device: int = 0
    
    # Memory
    pin_memory: bool = True
    non_blocking: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Performance
    num_threads: Optional[int] = None


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.inference = InferenceConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'batch_size'):
            self.training.batch_size = args.batch_size
        if hasattr(args, 'num_epochs'):
            self.training.num_epochs = args.num_epochs
        if hasattr(args, 'lr_g'):
            self.training.lr_g = args.lr_g
        if hasattr(args, 'lr_d'):
            self.training.lr_d = args.lr_d
        if hasattr(args, 'device'):
            self.system.device = args.device
        if hasattr(args, 'data_dir'):
            self.data.data_dir = args.data_dir
        if hasattr(args, 'log_dir'):
            self.logging.log_dir = args.log_dir
        if hasattr(args, 'checkpoint_dir'):
            self.logging.checkpoint_dir = args.checkpoint_dir
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.logging.log_dir,
            self.logging.checkpoint_dir,
            self.logging.sample_dir,
            self.inference.output_dir,
            os.path.dirname(self.data.data_dir)
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def print_config(self):
        """Print current configuration"""
        print("=== StackGAN Configuration ===")
        print(f"Model: vocab_size={self.model.vocab_size}, noise_dim={self.model.noise_dim}")
        print(f"Training: batch_size={self.training.batch_size}, epochs={self.training.num_epochs}")
        print(f"Data: data_dir={self.data.data_dir}")
        print(f"System: device={self.system.device}")
        print(f"Logging: log_dir={self.logging.log_dir}")
        print("=" * 30)


# Default configuration
config = Config()


def get_config():
    """Get the default configuration"""
    return config


def update_config(**kwargs):
    """Update configuration with keyword arguments"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to update nested configs
            for attr_name in dir(config):
                attr = getattr(config, attr_name)
                if hasattr(attr, key):
                    setattr(attr, key, value)
                    break


# Example usage:
# config = get_config()
# config.training.batch_size = 32
# config.model.noise_dim = 128
# config.print_config()