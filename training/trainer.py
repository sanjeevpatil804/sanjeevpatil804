import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
import matplotlib.pyplot as plt


class StackGANTrainer:
    """Trainer class for StackGAN text-to-image generation"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 lr_g=0.0002, lr_d=0.0002, beta1=0.5, beta2=0.999,
                 lambda_kl=2.0, lambda_gp=10.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizers
        self.optimizer_g_stage1 = optim.Adam(
            list(self.model.stage1_generator.parameters()) + 
            list(self.model.text_encoder.parameters()),
            lr=lr_g, betas=(beta1, beta2)
        )
        self.optimizer_d_stage1 = optim.Adam(
            self.model.stage1_discriminator.parameters(),
            lr=lr_d, betas=(beta1, beta2)
        )
        
        self.optimizer_g_stage2 = optim.Adam(
            list(self.model.stage2_generator.parameters()) + 
            list(self.model.text_encoder.parameters()),
            lr=lr_g, betas=(beta1, beta2)
        )
        self.optimizer_d_stage2 = optim.Adam(
            self.model.stage2_discriminator.parameters(),
            lr=lr_d, betas=(beta1, beta2)
        )
        
        # Loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_l1 = nn.L1Loss()
        
        # Hyperparameters
        self.lambda_kl = lambda_kl
        self.lambda_gp = lambda_gp
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Logging
        self.writer = None
        self.log_dir = None
        
    def setup_logging(self, log_dir='logs'):
        """Setup tensorboard logging"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, text_features):
        """Compute gradient penalty for WGAN-GP"""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        d_interpolates = discriminator(interpolates, text_features)
        
        fake = torch.ones(d_interpolates.size()).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_stage1(self, batch):
        """Train Stage-I Generator and Discriminator"""
        real_images = batch['images'].to(self.device)
        captions = batch['captions'].to(self.device)
        batch_size = real_images.size(0)
        
        # Resize images to 64x64 for Stage-I
        real_images_64 = torch.nn.functional.interpolate(real_images, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Generate noise
        noise = torch.randn(batch_size, 100).to(self.device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_d_stage1.zero_grad()
        
        # Real images
        real_validity = self.model.discriminate_stage1(real_images_64, captions)
        d_real_loss = self.criterion_gan(real_validity, torch.ones_like(real_validity))
        
        # Fake images
        with torch.no_grad():
            fake_images_64 = self.model.stage1_generator(noise, self.model.text_encoder(captions))
        fake_validity = self.model.discriminate_stage1(fake_images_64.detach(), captions)
        d_fake_loss = self.criterion_gan(fake_validity, torch.zeros_like(fake_validity))
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(
            self.model.stage1_discriminator, 
            real_images_64, 
            fake_images_64.detach(),
            self.model.text_encoder(captions)
        )
        
        d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp
        d_loss.backward()
        self.optimizer_d_stage1.step()
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_g_stage1.zero_grad()
        
        # Generate fake images
        fake_images_64 = self.model.stage1_generator(noise, self.model.text_encoder(captions))
        fake_validity = self.model.discriminate_stage1(fake_images_64, captions)
        
        # Generator loss
        g_loss = self.criterion_gan(fake_validity, torch.ones_like(fake_validity))
        
        # Feature matching loss
        real_features = self.model.stage1_discriminator.image_encoder(real_images_64)
        fake_features = self.model.stage1_discriminator.image_encoder(fake_images_64)
        feature_loss = self.criterion_l1(fake_features, real_features.detach())
        
        total_g_loss = g_loss + feature_loss
        total_g_loss.backward()
        self.optimizer_g_stage1.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': total_g_loss.item(),
            'real_validity': real_validity.mean().item(),
            'fake_validity': fake_validity.mean().item(),
            'feature_loss': feature_loss.item()
        }
    
    def train_stage2(self, batch):
        """Train Stage-II Generator and Discriminator"""
        real_images = batch['images'].to(self.device)
        captions = batch['captions'].to(self.device)
        batch_size = real_images.size(0)
        
        # Generate noise
        noise = torch.randn(batch_size, 100).to(self.device)
        
        # Generate Stage-I images
        with torch.no_grad():
            stage1_images = self.model.stage1_generator(noise, self.model.text_encoder(captions))
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_d_stage2.zero_grad()
        
        # Real images
        real_validity = self.model.discriminate_stage2(real_images, captions)
        d_real_loss = self.criterion_gan(real_validity, torch.ones_like(real_validity))
        
        # Fake images
        with torch.no_grad():
            fake_images = self.model.stage2_generator(stage1_images, noise, self.model.text_encoder(captions))
        fake_validity = self.model.discriminate_stage2(fake_images.detach(), captions)
        d_fake_loss = self.criterion_gan(fake_validity, torch.zeros_like(fake_validity))
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(
            self.model.stage2_discriminator,
            real_images,
            fake_images.detach(),
            self.model.text_encoder(captions)
        )
        
        d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gp
        d_loss.backward()
        self.optimizer_d_stage2.step()
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_g_stage2.zero_grad()
        
        # Generate fake images
        fake_images = self.model.stage2_generator(stage1_images, noise, self.model.text_encoder(captions))
        fake_validity = self.model.discriminate_stage2(fake_images, captions)
        
        # Generator loss
        g_loss = self.criterion_gan(fake_validity, torch.ones_like(fake_validity))
        
        # Feature matching loss
        real_features = self.model.stage2_discriminator.image_encoder(real_images)
        fake_features = self.model.stage2_discriminator.image_encoder(fake_images)
        feature_loss = self.criterion_l1(fake_features, real_features.detach())
        
        total_g_loss = g_loss + feature_loss
        total_g_loss.backward()
        self.optimizer_g_stage2.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': total_g_loss.item(),
            'real_validity': real_validity.mean().item(),
            'fake_validity': fake_validity.mean().item(),
            'feature_loss': feature_loss.item()
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        stage1_losses = {'d_loss': [], 'g_loss': [], 'real_validity': [], 'fake_validity': [], 'feature_loss': []}
        stage2_losses = {'d_loss': [], 'g_loss': [], 'real_validity': [], 'fake_validity': [], 'feature_loss': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Train Stage-I
            stage1_metrics = self.train_stage1(batch)
            for key, value in stage1_metrics.items():
                stage1_losses[key].append(value)
            
            # Train Stage-II
            stage2_metrics = self.train_stage2(batch)
            for key, value in stage2_metrics.items():
                stage2_losses[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'S1_D': f"{stage1_metrics['d_loss']:.3f}",
                'S1_G': f"{stage1_metrics['g_loss']:.3f}",
                'S2_D': f"{stage2_metrics['d_loss']:.3f}",
                'S2_G': f"{stage2_metrics['g_loss']:.3f}"
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % 100 == 0:
                for key, value in stage1_metrics.items():
                    self.writer.add_scalar(f'Stage1/{key}', value, self.global_step)
                for key, value in stage2_metrics.items():
                    self.writer.add_scalar(f'Stage2/{key}', value, self.global_step)
                self.global_step += 1
        
        # Return average losses
        stage1_avg = {key: np.mean(values) for key, values in stage1_losses.items()}
        stage2_avg = {key: np.mean(values) for key, values in stage2_losses.items()}
        
        return stage1_avg, stage2_avg
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        val_losses = {'stage1_d': [], 'stage1_g': [], 'stage2_d': [], 'stage2_g': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                real_images = batch['images'].to(self.device)
                captions = batch['captions'].to(self.device)
                batch_size = real_images.size(0)
                
                # Generate noise
                noise = torch.randn(batch_size, 100).to(self.device)
                
                # Stage-I validation
                real_images_64 = torch.nn.functional.interpolate(real_images, size=(64, 64), mode='bilinear', align_corners=False)
                fake_images_64 = self.model.stage1_generator(noise, self.model.text_encoder(captions))
                
                real_validity_s1 = self.model.discriminate_stage1(real_images_64, captions)
                fake_validity_s1 = self.model.discriminate_stage1(fake_images_64, captions)
                
                d_loss_s1 = self.criterion_gan(real_validity_s1, torch.ones_like(real_validity_s1)) + \
                           self.criterion_gan(fake_validity_s1, torch.zeros_like(fake_validity_s1))
                g_loss_s1 = self.criterion_gan(fake_validity_s1, torch.ones_like(fake_validity_s1))
                
                # Stage-II validation
                fake_images = self.model.stage2_generator(fake_images_64, noise, self.model.text_encoder(captions))
                
                real_validity_s2 = self.model.discriminate_stage2(real_images, captions)
                fake_validity_s2 = self.model.discriminate_stage2(fake_images, captions)
                
                d_loss_s2 = self.criterion_gan(real_validity_s2, torch.ones_like(real_validity_s2)) + \
                           self.criterion_gan(fake_validity_s2, torch.zeros_like(fake_validity_s2))
                g_loss_s2 = self.criterion_gan(fake_validity_s2, torch.ones_like(fake_validity_s2))
                
                val_losses['stage1_d'].append(d_loss_s1.item())
                val_losses['stage1_g'].append(g_loss_s1.item())
                val_losses['stage2_d'].append(d_loss_s2.item())
                val_losses['stage2_g'].append(g_loss_s2.item())
        
        return {key: np.mean(values) for key, values in val_losses.items()}
    
    def save_samples(self, epoch, num_samples=8):
        """Save generated samples"""
        self.model.eval()
        
        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        captions = batch['captions'][:num_samples].to(self.device)
        real_images = batch['images'][:num_samples].to(self.device)
        
        with torch.no_grad():
            noise = torch.randn(num_samples, 100).to(self.device)
            
            # Generate Stage-I images
            stage1_images = self.model.stage1_generator(noise, self.model.text_encoder(captions))
            
            # Generate Stage-II images
            stage2_images = self.model.stage2_generator(stage1_images, noise, self.model.text_encoder(captions))
            
            # Save images
            if self.log_dir:
                sample_dir = os.path.join(self.log_dir, 'samples', f'epoch_{epoch}')
                os.makedirs(sample_dir, exist_ok=True)
                
                for i in range(num_samples):
                    # Save real image
                    real_img = real_images[i].cpu().permute(1, 2, 0).numpy()
                    real_img = (real_img + 1) / 2  # Denormalize
                    plt.imsave(os.path.join(sample_dir, f'real_{i}.png'), real_img)
                    
                    # Save Stage-I image
                    stage1_img = stage1_images[i].cpu().permute(1, 2, 0).numpy()
                    stage1_img = (stage1_img + 1) / 2  # Denormalize
                    plt.imsave(os.path.join(sample_dir, f'stage1_{i}.png'), stage1_img)
                    
                    # Save Stage-II image
                    stage2_img = stage2_images[i].cpu().permute(1, 2, 0).numpy()
                    stage2_img = (stage2_img + 1) / 2  # Denormalize
                    plt.imsave(os.path.join(sample_dir, f'stage2_{i}.png'), stage2_img)
    
    def save_checkpoint(self, epoch, save_dir='checkpoints'):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_stage1_state_dict': self.optimizer_g_stage1.state_dict(),
            'optimizer_d_stage1_state_dict': self.optimizer_d_stage1.state_dict(),
            'optimizer_g_stage2_state_dict': self.optimizer_g_stage2.state_dict(),
            'optimizer_d_stage2_state_dict': self.optimizer_d_stage2.state_dict(),
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f'stackgan_epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g_stage1.load_state_dict(checkpoint['optimizer_g_stage1_state_dict'])
        self.optimizer_d_stage1.load_state_dict(checkpoint['optimizer_d_stage1_state_dict'])
        self.optimizer_g_stage2.load_state_dict(checkpoint['optimizer_g_stage2_state_dict'])
        self.optimizer_d_stage2.load_state_dict(checkpoint['optimizer_d_stage2_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs, save_interval=10, sample_interval=5):
        """Main training loop"""
        print("Starting StackGAN training...")
        
        for epoch in range(self.current_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            stage1_losses, stage2_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Log results
            print(f"Stage-I - D: {stage1_losses['d_loss']:.4f}, G: {stage1_losses['g_loss']:.4f}")
            print(f"Stage-II - D: {stage2_losses['d_loss']:.4f}, G: {stage2_losses['g_loss']:.4f}")
            print(f"Validation - S1_D: {val_losses['stage1_d']:.4f}, S1_G: {val_losses['stage1_g']:.4f}")
            print(f"Validation - S2_D: {val_losses['stage2_d']:.4f}, S2_G: {val_losses['stage2_g']:.4f}")
            
            # Save samples
            if (epoch + 1) % sample_interval == 0:
                self.save_samples(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            self.current_epoch = epoch + 1
        
        print("Training completed!")