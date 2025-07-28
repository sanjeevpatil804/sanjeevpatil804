import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Tuple
import argparse
import pickle
import time


class StackGANInference:
    """Inference class for StackGAN text-to-image generation"""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.vocab = None
        
        # Load model and vocabulary
        self.load_model(model_path)
        self.load_vocabulary(vocab_path)
        
    def load_model(self, model_path: str):
        """Load trained StackGAN model"""
        from models.stackgan import StackGAN
        
        # Initialize model with same architecture as training
        vocab_size = 10000  # This should match your training vocabulary size
        self.model = StackGAN(
            vocab_size=vocab_size,
            embed_dim=256,
            hidden_dim=256,
            noise_dim=100,
            gf_dim=128,
            df_dim=128,
            ef_dim=128
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def load_vocabulary(self, vocab_path: str):
        """Load vocabulary for text processing"""
        from data.dataset import Vocabulary
        
        self.vocab = Vocabulary()
        
        # Load vocabulary from file
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.vocab.word2idx = vocab_data['word2idx']
                self.vocab.idx2word = vocab_data['idx2word']
                self.vocab.word_freq = vocab_data['word_freq']
        else:
            print(f"Warning: Vocabulary file {vocab_path} not found. Using default vocabulary.")
    
    def preprocess_text(self, text: str, max_length: int = 50) -> torch.Tensor:
        """Preprocess text input"""
        if self.vocab is None:
            raise ValueError("Vocabulary not loaded. Please load vocabulary first.")
        
        return self.vocab.encode(text, max_length)
    
    def generate_image(self, text: str, noise: Optional[torch.Tensor] = None, 
                      temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate image from text description"""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess text
            text_tensor = self.preprocess_text(text).unsqueeze(0).to(self.device)
            
            # Generate noise if not provided
            if noise is None:
                noise = torch.randn(1, 100).to(self.device)
            
            # Apply temperature to noise
            if temperature != 1.0:
                noise = noise * temperature
            
            # Generate Stage-I image (64x64)
            stage1_image = self.model.stage1_generator(noise, self.model.text_encoder(text_tensor))
            
            # Generate Stage-II image (256x256)
            stage2_image = self.model.stage2_generator(stage1_image, noise, self.model.text_encoder(text_tensor))
            
            return stage1_image, stage2_image
    
    def generate_multiple_images(self, text: str, num_images: int = 4, 
                               temperature: float = 1.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate multiple images from the same text description"""
        results = []
        
        for i in range(num_images):
            noise = torch.randn(1, 100).to(self.device)
            stage1, stage2 = self.generate_image(text, noise, temperature)
            results.append((stage1, stage2))
        
        return results
    
    def save_image(self, image_tensor: torch.Tensor, save_path: str):
        """Save image tensor to file"""
        # Convert tensor to numpy array
        image = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        image = (image + 1) / 2
        image = np.clip(image, 0, 1)
        
        # Convert to PIL Image and save
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_pil.save(save_path)
    
    def create_comparison_grid(self, stage1_images: List[torch.Tensor], 
                             stage2_images: List[torch.Tensor], 
                             save_path: str, text: str = ""):
        """Create a comparison grid of Stage-I and Stage-II images"""
        num_images = len(stage1_images)
        fig, axes = plt.subplots(2, num_images, figsize=(4*num_images, 8))
        
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_images):
            # Stage-I image
            stage1_img = stage1_images[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
            stage1_img = (stage1_img + 1) / 2
            axes[0, i].imshow(stage1_img)
            axes[0, i].set_title(f'Stage-I {i+1}')
            axes[0, i].axis('off')
            
            # Stage-II image
            stage2_img = stage2_images[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
            stage2_img = (stage2_img + 1) / 2
            axes[1, i].imshow(stage2_img)
            axes[1, i].set_title(f'Stage-II {i+1}')
            axes[1, i].axis('off')
        
        if text:
            fig.suptitle(f'Text: "{text}"', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def interactive_generation(self):
        """Interactive text-to-image generation"""
        print("Interactive StackGAN Text-to-Image Generation")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text description: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                print("Please enter a text description.")
                continue
            
            try:
                # Generate image
                stage1, stage2 = self.generate_image(text)
                
                # Save images
                os.makedirs('generated_images', exist_ok=True)
                timestamp = int(time.time())
                
                self.save_image(stage1, f'generated_images/stage1_{timestamp}.png')
                self.save_image(stage2, f'generated_images/stage2_{timestamp}.png')
                
                print(f"Generated images saved as:")
                print(f"  Stage-I: generated_images/stage1_{timestamp}.png")
                print(f"  Stage-II: generated_images/stage2_{timestamp}.png")
                
            except Exception as e:
                print(f"Error generating image: {e}")


def main():
    parser = argparse.ArgumentParser(description='StackGAN Text-to-Image Generation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--text', type=str, help='Text description for image generation')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='generated_images', help='Output directory')
    parser.add_argument('--temperature', type=float, default=1.0, help='Noise temperature for generation')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize inference
    generator = StackGANInference(args.model_path, args.vocab_path, args.device)
    
    if args.interactive:
        generator.interactive_generation()
    else:
        if not args.text:
            print("Please provide text description or use --interactive mode")
            return
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate images
        print(f"Generating {args.num_images} images for text: '{args.text}'")
        
        results = generator.generate_multiple_images(args.text, args.num_images, args.temperature)
        
        # Save individual images
        for i, (stage1, stage2) in enumerate(results):
            generator.save_image(stage1, os.path.join(args.output_dir, f'stage1_{i+1}.png'))
            generator.save_image(stage2, os.path.join(args.output_dir, f'stage2_{i+1}.png'))
        
        # Create comparison grid
        stage1_images = [result[0] for result in results]
        stage2_images = [result[1] for result in results]
        
        generator.create_comparison_grid(
            stage1_images, 
            stage2_images, 
            os.path.join(args.output_dir, 'comparison_grid.png'),
            args.text
        )
        
        print(f"Generated images saved in {args.output_dir}/")


if __name__ == "__main__":
    main()