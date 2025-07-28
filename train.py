#!/usr/bin/env python3
"""
StackGAN Text-to-Image Generation Training Script
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle

from models.stackgan import StackGAN
from data.dataset import CUBDataset, TextImageDataset, get_transforms, create_dataloader, create_sample_data
from training.trainer import StackGANTrainer


def setup_data(data_dir, batch_size=16, num_workers=4, create_sample=False):
    """Setup data loaders"""
    if create_sample:
        print("Creating sample data...")
        create_sample_data(data_dir, num_samples=1000)
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size=256)
    
    # Create datasets
    try:
        # Try CUB dataset first
        train_dataset = CUBDataset(data_dir, split='train', transform=train_transform)
        val_dataset = CUBDataset(data_dir, split='val', transform=val_transform)
        print("Using CUB dataset")
    except:
        # Fall back to generic dataset
        train_dataset = TextImageDataset(data_dir, split='train', transform=train_transform)
        val_dataset = TextImageDataset(data_dir, split='val', transform=val_transform)
        print("Using generic text-image dataset")
    
    # Create data loaders
    train_loader = create_dataloader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    # Save vocabulary
    vocab = train_dataset.vocab
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({
            'word2idx': vocab.word2idx,
            'idx2word': vocab.idx2word,
            'word_freq': vocab.word_freq
        }, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    return train_loader, val_loader, vocab


def setup_model(vocab_size, device):
    """Setup StackGAN model"""
    model = StackGAN(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=256,
        noise_dim=100,
        gf_dim=128,
        df_dim=128,
        ef_dim=128
    )
    
    print(f"Model created with vocabulary size: {vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train StackGAN for text-to-image generation')
    parser.add_argument('--data_dir', type=str, default='data/cub', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Learning rate for generators')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for discriminators')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=5, help='Save samples every N epochs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logging')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--create_sample', action='store_true', help='Create sample data if dataset not found')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup data
    print("Setting up data...")
    train_loader, val_loader, vocab = setup_data(
        args.data_dir, 
        args.batch_size, 
        args.num_workers,
        args.create_sample
    )
    
    # Setup model
    print("Setting up model...")
    model = setup_model(len(vocab), args.device)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = StackGANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr_g=args.lr_g,
        lr_d=args.lr_d
    )
    
    # Setup logging
    trainer.setup_logging(args.log_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Print training info
    print(f"\nTraining Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Learning rate (G): {args.lr_g}")
    print(f"  Learning rate (D): {args.lr_d}")
    print(f"  Device: {args.device}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    
    # Start training
    print("\nStarting training...")
    trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()