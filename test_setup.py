#!/usr/bin/env python3
"""
Test script for StackGAN setup
"""

import torch
import numpy as np
import os
import sys

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.stackgan import StackGAN
        print("✓ StackGAN model imported successfully")
    except Exception as e:
        print(f"✗ Failed to import StackGAN: {e}")
        return False
    
    try:
        from data.dataset import Vocabulary, TextImageDataset, create_sample_data
        print("✓ Dataset modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import dataset modules: {e}")
        return False
    
    try:
        from training.trainer import StackGANTrainer
        print("✓ Trainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import trainer: {e}")
        return False
    
    try:
        from inference.generator import StackGANInference
        print("✓ Inference module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import inference: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        from models.stackgan import StackGAN
        
        # Create model
        model = StackGAN(
            vocab_size=1000,
            embed_dim=256,
            hidden_dim=256,
            noise_dim=100,
            gf_dim=128,
            df_dim=128,
            ef_dim=128
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        text = torch.randint(0, 1000, (batch_size, 50))
        noise = torch.randn(batch_size, 100)
        
        stage1, stage2 = model(text, noise)
        
        print(f"✓ Forward pass successful")
        print(f"  Stage-I output shape: {stage1.shape}")
        print(f"  Stage-II output shape: {stage2.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_dataset():
    """Test dataset creation and loading"""
    print("\nTesting dataset...")
    
    try:
        from data.dataset import create_sample_data, TextImageDataset, get_transforms
        
        # Create sample data
        data_dir = "test_data"
        create_sample_data(data_dir, num_samples=10)
        print("✓ Sample data created successfully")
        
        # Create dataset
        transform, _ = get_transforms(image_size=256)
        dataset = TextImageDataset(data_dir, transform=transform)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Caption shape: {sample['caption'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_vocabulary():
    """Test vocabulary creation and text processing"""
    print("\nTesting vocabulary...")
    
    try:
        from data.dataset import Vocabulary
        
        # Create vocabulary
        vocab = Vocabulary(min_freq=1)
        
        # Sample texts
        texts = [
            "A beautiful red flower in a garden",
            "A majestic mountain landscape at sunset",
            "A cute cat sitting on a windowsill"
        ]
        
        # Build vocabulary
        vocab.build_vocab(texts)
        print(f"✓ Vocabulary built with {len(vocab)} words")
        
        # Test encoding
        text = "A beautiful red flower"
        encoded = vocab.encode(text, max_length=20)
        print(f"✓ Text encoding successful: {encoded.shape}")
        
        # Test decoding
        decoded = vocab.decode(encoded)
        print(f"✓ Text decoding successful: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Vocabulary test failed: {e}")
        return False


def test_device():
    """Test device availability"""
    print("\nTesting device...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✓ CPU only mode")
    
    return True


def test_dependencies():
    """Test if all required dependencies are available"""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('tqdm', 'tqdm'),
        ('nltk', 'NLTK'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('accelerate', 'Accelerate'),
        ('wandb', 'Weights & Biases'),
        ('tensorboard', 'TensorBoard')
    ]
    
    all_available = True
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name} available")
        except ImportError:
            print(f"✗ {display_name} not available")
            all_available = False
    
    return all_available


def main():
    """Run all tests"""
    print("=== StackGAN Setup Test ===\n")
    
    tests = [
        test_dependencies,
        test_device,
        test_imports,
        test_vocabulary,
        test_dataset,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! StackGAN is ready to use.")
        print("\nNext steps:")
        print("1. Run demo: python demo.py --mode full")
        print("2. Train model: python train.py --create_sample --num_epochs 10")
        print("3. Generate images: python inference/generator.py --interactive")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())