#!/usr/bin/env python3
"""
StackGAN Text-to-Image Generation Demo
This script demonstrates how to use the StackGAN model for text-to-image generation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from models.stackgan import StackGAN
from data.dataset import create_sample_data, Vocabulary
from training.trainer import StackGANTrainer
from data.dataset import get_transforms, create_dataloader


def create_demo_data(data_dir='demo_data', num_samples=100):
    """Create demo data for testing"""
    print(f"Creating demo data in {data_dir}...")
    create_sample_data(data_dir, num_samples=num_samples)
    print("Demo data created successfully!")


def setup_demo_model(vocab_size=1000):
    """Setup a demo StackGAN model"""
    model = StackGAN(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=256,
        noise_dim=100,
        gf_dim=128,
        df_dim=128,
        ef_dim=128
    )
    return model


def demo_training(data_dir='demo_data', num_epochs=5, batch_size=8):
    """Demo training process"""
    print("Starting demo training...")
    
    # Setup data
    train_transform, val_transform = get_transforms(image_size=256)
    
    from data.dataset import TextImageDataset
    train_dataset = TextImageDataset(data_dir, split='train', transform=train_transform)
    val_dataset = TextImageDataset(data_dir, split='train', transform=val_transform)  # Use same data for demo
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Setup model
    model = setup_demo_model(len(train_dataset.vocab))
    
    # Setup trainer
    trainer = StackGANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr_g=0.0002,
        lr_d=0.0002
    )
    
    # Setup logging
    trainer.setup_logging('demo_logs')
    
    # Train for a few epochs
    trainer.train(num_epochs=num_epochs, save_interval=num_epochs, sample_interval=2)
    
    return model, train_dataset.vocab


def demo_generation(model, vocab, text_descriptions, output_dir='demo_output'):
    """Demo image generation from text"""
    print("Starting demo generation...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    for i, text in enumerate(text_descriptions):
        print(f"Generating image for: '{text}'")
        
        # Encode text
        text_encoded = vocab.encode(text, max_length=50).unsqueeze(0).to(device)
        
        # Generate noise
        noise = torch.randn(1, 100).to(device)
        
        with torch.no_grad():
            # Generate Stage-I image
            stage1_image = model.stage1_generator(noise, model.text_encoder(text_encoded))
            
            # Generate Stage-II image
            stage2_image = model.stage2_generator(stage1_image, noise, model.text_encoder(text_encoded))
            
            # Save images
            stage1_img = stage1_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            stage1_img = (stage1_img + 1) / 2  # Denormalize
            stage1_img = np.clip(stage1_img, 0, 1)
            
            stage2_img = stage2_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            stage2_img = (stage2_img + 1) / 2  # Denormalize
            stage2_img = np.clip(stage2_img, 0, 1)
            
            # Save as PIL images
            stage1_pil = Image.fromarray((stage1_img * 255).astype(np.uint8))
            stage2_pil = Image.fromarray((stage2_img * 255).astype(np.uint8))
            
            stage1_pil.save(os.path.join(output_dir, f'stage1_{i+1}.png'))
            stage2_pil.save(os.path.join(output_dir, f'stage2_{i+1}.png'))
            
            # Create comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(stage1_img)
            ax1.set_title('Stage-I (64x64)')
            ax1.axis('off')
            
            ax2.imshow(stage2_img)
            ax2.set_title('Stage-II (256x256)')
            ax2.axis('off')
            
            plt.suptitle(f'Text: "{text}"', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{i+1}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: stage1_{i+1}.png, stage2_{i+1}.png, comparison_{i+1}.png")
    
    print(f"All generated images saved in {output_dir}/")


def interactive_demo(model, vocab):
    """Interactive demo for text-to-image generation"""
    print("\nInteractive StackGAN Demo")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    device = next(model.parameters()).device
    
    while True:
        text = input("\nEnter text description: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            print("Please enter a text description.")
            continue
        
        try:
            # Encode text
            text_encoded = vocab.encode(text, max_length=50).unsqueeze(0).to(device)
            
            # Generate noise
            noise = torch.randn(1, 100).to(device)
            
            with torch.no_grad():
                # Generate Stage-I image
                stage1_image = model.stage1_generator(noise, model.text_encoder(text_encoded))
                
                # Generate Stage-II image
                stage2_image = model.stage2_generator(stage1_image, noise, model.text_encoder(text_encoded))
                
                # Display images
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                stage1_img = stage1_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
                stage1_img = (stage1_img + 1) / 2
                ax1.imshow(stage1_img)
                ax1.set_title('Stage-I (64x64)')
                ax1.axis('off')
                
                stage2_img = stage2_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
                stage2_img = (stage2_img + 1) / 2
                ax2.imshow(stage2_img)
                ax2.set_title('Stage-II (256x256)')
                ax2.axis('off')
                
                plt.suptitle(f'Text: "{text}"', fontsize=14)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error generating image: {e}")


def main():
    parser = argparse.ArgumentParser(description='StackGAN Text-to-Image Generation Demo')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['data', 'train', 'generate', 'interactive', 'full'],
                       help='Demo mode')
    parser.add_argument('--data_dir', type=str, default='demo_data', help='Data directory')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='demo_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Sample text descriptions for generation
    sample_texts = [
        "A beautiful red flower in a garden",
        "A majestic mountain landscape at sunset",
        "A cute cat sitting on a windowsill",
        "A modern city skyline at night",
        "A peaceful lake surrounded by trees"
    ]
    
    if args.mode == 'data':
        # Create demo data only
        create_demo_data(args.data_dir)
        
    elif args.mode == 'train':
        # Train model only
        model, vocab = demo_training(args.data_dir, args.num_epochs, args.batch_size)
        
    elif args.mode == 'generate':
        # Generate images only (requires trained model)
        print("Loading pre-trained model...")
        model = setup_demo_model()
        # Note: In a real scenario, you would load a trained model here
        print("Demo generation requires a trained model. Please run training first.")
        
    elif args.mode == 'interactive':
        # Interactive demo only
        print("Loading pre-trained model...")
        model = setup_demo_model()
        vocab = Vocabulary()
        interactive_demo(model, vocab)
        
    elif args.mode == 'full':
        # Full demo: create data, train, and generate
        print("=== StackGAN Text-to-Image Generation Demo ===\n")
        
        # Step 1: Create demo data
        print("Step 1: Creating demo data...")
        create_demo_data(args.data_dir)
        
        # Step 2: Train model
        print("\nStep 2: Training model...")
        model, vocab = demo_training(args.data_dir, args.num_epochs, args.batch_size)
        
        # Step 3: Generate images
        print("\nStep 3: Generating images...")
        demo_generation(model, vocab, sample_texts, args.output_dir)
        
        # Step 4: Interactive demo
        print("\nStep 4: Interactive demo...")
        interactive_demo(model, vocab)
        
        print("\nDemo completed! Check the output directory for generated images.")
    
    else:
        print("Invalid mode. Please choose from: data, train, generate, interactive, full")


if __name__ == "__main__":
    main()