import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization for GANs"""
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ResBlock(nn.Module):
    """Residual Block for Generator"""
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class TextEncoder(nn.Module):
    """Text Encoder using LSTM"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, text):
        embedded = self.embed(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last hidden state from both directions
        text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return text_features


class Stage1Generator(nn.Module):
    """Stage-I Generator: Text to 64x64 images"""
    def __init__(self, text_dim=256, noise_dim=100, gf_dim=128, ef_dim=128):
        super().__init__()
        self.text_dim = text_dim
        self.noise_dim = noise_dim
        self.gf_dim = gf_dim
        self.ef_dim = ef_dim
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, ef_dim * 4),
            nn.BatchNorm1d(ef_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(ef_dim * 4, ef_dim * 2),
            nn.BatchNorm1d(ef_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        # Noise processing
        self.noise_net = nn.Sequential(
            nn.Linear(noise_dim, gf_dim * 8),
            nn.BatchNorm1d(gf_dim * 8),
            nn.ReLU(inplace=True)
        )
        
        # Generator blocks
        self.g_net = nn.Sequential(
            # Input: (batch_size, gf_dim*8 + ef_dim*2, 1, 1)
            nn.ConvTranspose2d(gf_dim * 8 + ef_dim * 2, gf_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gf_dim * 8),
            nn.ReLU(inplace=True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 4),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 2),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(gf_dim * 2, gf_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(gf_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, noise, text_features):
        # Process text features
        text_encoded = self.text_encoder(text_features)
        text_encoded = text_encoded.unsqueeze(2).unsqueeze(3)  # (batch_size, ef_dim*2, 1, 1)
        
        # Process noise
        noise_encoded = self.noise_net(noise)
        noise_encoded = noise_encoded.unsqueeze(2).unsqueeze(3)  # (batch_size, gf_dim*8, 1, 1)
        
        # Concatenate noise and text features
        combined = torch.cat([noise_encoded, text_encoded], dim=1)
        
        # Generate image
        fake_images = self.g_net(combined)
        return fake_images


class Stage1Discriminator(nn.Module):
    """Stage-I Discriminator: 64x64 images + text"""
    def __init__(self, text_dim=256, df_dim=128, ef_dim=128):
        super().__init__()
        self.text_dim = text_dim
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, ef_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ef_dim * 4, ef_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(df_dim, df_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(df_dim * 8 + ef_dim * 2, df_dim * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, images, text_features):
        # Encode images
        img_features = self.image_encoder(images)  # (batch_size, df_dim*8, 4, 4)
        
        # Encode text
        text_encoded = self.text_encoder(text_features)  # (batch_size, ef_dim*2)
        text_encoded = text_encoded.unsqueeze(2).unsqueeze(3)  # (batch_size, ef_dim*2, 1, 1)
        text_encoded = text_encoded.expand(-1, -1, 4, 4)  # (batch_size, ef_dim*2, 4, 4)
        
        # Concatenate image and text features
        combined = torch.cat([img_features, text_encoded], dim=1)
        
        # Classify
        validity = self.classifier(combined)
        return validity.squeeze()


class Stage2Generator(nn.Module):
    """Stage-II Generator: 64x64 + text to 256x256 images"""
    def __init__(self, text_dim=256, noise_dim=100, gf_dim=128, ef_dim=128):
        super().__init__()
        self.text_dim = text_dim
        self.noise_dim = noise_dim
        self.gf_dim = gf_dim
        self.ef_dim = ef_dim
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, ef_dim * 4),
            nn.BatchNorm1d(ef_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(ef_dim * 4, ef_dim * 2),
            nn.BatchNorm1d(ef_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        # Noise processing
        self.noise_net = nn.Sequential(
            nn.Linear(noise_dim, gf_dim * 8),
            nn.BatchNorm1d(gf_dim * 8),
            nn.ReLU(inplace=True)
        )
        
        # Stage-I image encoder
        self.stage1_encoder = nn.Sequential(
            nn.Conv2d(3, gf_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(gf_dim, gf_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        # Generator blocks
        self.g_net = nn.Sequential(
            # Input: (batch_size, gf_dim*8 + ef_dim*2 + gf_dim*2, 64, 64)
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(gf_dim * 8 + ef_dim * 2 + gf_dim * 2, gf_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 4),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 2),
            nn.ReLU(inplace=True),
            
            # Final convolution
            nn.Conv2d(gf_dim * 2, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, stage1_images, noise, text_features):
        # Process text features
        text_encoded = self.text_encoder(text_features)
        text_encoded = text_encoded.unsqueeze(2).unsqueeze(3)  # (batch_size, ef_dim*2, 1, 1)
        text_encoded = text_encoded.expand(-1, -1, 64, 64)  # (batch_size, ef_dim*2, 64, 64)
        
        # Process noise
        noise_encoded = self.noise_net(noise)
        noise_encoded = noise_encoded.unsqueeze(2).unsqueeze(3)  # (batch_size, gf_dim*8, 1, 1)
        noise_encoded = noise_encoded.expand(-1, -1, 64, 64)  # (batch_size, gf_dim*8, 64, 64)
        
        # Encode stage-I images
        stage1_encoded = self.stage1_encoder(stage1_images)  # (batch_size, gf_dim*2, 64, 64)
        
        # Concatenate all features
        combined = torch.cat([noise_encoded, text_encoded, stage1_encoded], dim=1)
        
        # Generate high-resolution image
        fake_images = self.g_net(combined)
        return fake_images


class Stage2Discriminator(nn.Module):
    """Stage-II Discriminator: 256x256 images + text"""
    def __init__(self, text_dim=256, df_dim=128, ef_dim=128):
        super().__init__()
        self.text_dim = text_dim
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, ef_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ef_dim * 4, ef_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(3, df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(df_dim, df_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(df_dim * 8, df_dim * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(df_dim * 16 + ef_dim * 2, df_dim * 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim * 16, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, images, text_features):
        # Encode images
        img_features = self.image_encoder(images)  # (batch_size, df_dim*16, 8, 8)
        
        # Encode text
        text_encoded = self.text_encoder(text_features)  # (batch_size, ef_dim*2)
        text_encoded = text_encoded.unsqueeze(2).unsqueeze(3)  # (batch_size, ef_dim*2, 1, 1)
        text_encoded = text_encoded.expand(-1, -1, 8, 8)  # (batch_size, ef_dim*2, 8, 8)
        
        # Concatenate image and text features
        combined = torch.cat([img_features, text_encoded], dim=1)
        
        # Classify
        validity = self.classifier(combined)
        return validity.squeeze()


class StackGAN(nn.Module):
    """Complete StackGAN model"""
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, noise_dim=100, 
                 gf_dim=128, df_dim=128, ef_dim=128):
        super().__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)
        text_dim = hidden_dim * 2  # Bidirectional LSTM
        
        # Stage-I components
        self.stage1_generator = Stage1Generator(text_dim, noise_dim, gf_dim, ef_dim)
        self.stage1_discriminator = Stage1Discriminator(text_dim, df_dim, ef_dim)
        
        # Stage-II components
        self.stage2_generator = Stage2Generator(text_dim, noise_dim, gf_dim, ef_dim)
        self.stage2_discriminator = Stage2Discriminator(text_dim, df_dim, ef_dim)
        
    def forward(self, text, noise):
        # Encode text
        text_features = self.text_encoder(text)
        
        # Stage-I generation
        stage1_images = self.stage1_generator(noise, text_features)
        
        # Stage-II generation
        stage2_images = self.stage2_generator(stage1_images, noise, text_features)
        
        return stage1_images, stage2_images
    
    def discriminate_stage1(self, images, text):
        text_features = self.text_encoder(text)
        return self.stage1_discriminator(images, text_features)
    
    def discriminate_stage2(self, images, text):
        text_features = self.text_encoder(text)
        return self.stage2_discriminator(images, text_features)