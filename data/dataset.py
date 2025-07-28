import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pickle


class Vocabulary:
    """Vocabulary class for text processing"""
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        """Build vocabulary from text corpus"""
        for text in texts:
            tokens = word_tokenize(text.lower())
            self.word_freq.update(tokens)
        
        # Add words that meet minimum frequency
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text, max_length=50):
        """Encode text to indices"""
        tokens = word_tokenize(text.lower())
        tokens = ['<START>'] + tokens[:max_length-2] + ['<END>']
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens += ['<PAD>'] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        """Decode indices back to text"""
        tokens = [self.idx2word[idx] for idx in indices]
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.word2idx)


class CUBDataset(Dataset):
    """CUB-200-2011 Dataset for text-to-image generation"""
    def __init__(self, root_dir, split='train', transform=None, max_length=50):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_length = max_length
        
        # Load image paths and captions
        self.images, self.captions = self._load_data()
        
        # Build vocabulary
        self.vocab = Vocabulary(min_freq=5)
        self.vocab.build_vocab(self.captions)
        
    def _load_data(self):
        """Load image paths and captions"""
        images = []
        captions = []
        
        # Load captions
        captions_file = os.path.join(self.root_dir, 'captions.txt')
        if os.path.exists(captions_file):
            with open(captions_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_path = parts[0]
                        caption = parts[1]
                        
                        # Check if image exists
                        full_img_path = os.path.join(self.root_dir, 'images', img_path)
                        if os.path.exists(full_img_path):
                            images.append(full_img_path)
                            captions.append(caption)
        
        return images, captions
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Encode caption
        caption = self.captions[idx]
        caption_encoded = self.vocab.encode(caption, self.max_length)
        
        return {
            'image': image,
            'caption': caption_encoded,
            'caption_text': caption,
            'image_path': img_path
        }


class TextImageDataset(Dataset):
    """Generic text-image dataset"""
    def __init__(self, data_dir, split='train', transform=None, max_length=50):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_length = max_length
        
        # Load data
        self.images, self.captions = self._load_data()
        
        # Build vocabulary
        self.vocab = Vocabulary(min_freq=5)
        self.vocab.build_vocab(self.captions)
        
    def _load_data(self):
        """Load image paths and captions from data directory"""
        images = []
        captions = []
        
        # Look for common data formats
        metadata_files = ['captions.json', 'annotations.json', 'metadata.json']
        
        for metadata_file in metadata_files:
            file_path = os.path.join(self.data_dir, metadata_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if 'image' in item and 'caption' in item:
                                img_path = os.path.join(self.data_dir, item['image'])
                                if os.path.exists(img_path):
                                    images.append(img_path)
                                    captions.append(item['caption'])
                    elif isinstance(data, dict):
                        for img_name, caption in data.items():
                            img_path = os.path.join(self.data_dir, img_name)
                            if os.path.exists(img_path):
                                images.append(img_path)
                                captions.append(caption)
                break
        
        return images, captions
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Encode caption
        caption = self.captions[idx]
        caption_encoded = self.vocab.encode(caption, self.max_length)
        
        return {
            'image': image,
            'caption': caption_encoded,
            'caption_text': caption,
            'image_path': img_path
        }


def get_transforms(image_size=256):
    """Get image transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4):
    """Create DataLoader for the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    captions = torch.stack([item['caption'] for item in batch])
    caption_texts = [item['caption_text'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'captions': captions,
        'caption_texts': caption_texts,
        'image_paths': image_paths
    }


def create_sample_data(data_dir, num_samples=1000):
    """Create sample data for testing"""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    
    # Create sample images (random noise)
    from PIL import Image
    import numpy as np
    
    captions = []
    for i in range(num_samples):
        # Create random image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(data_dir, 'images', f'sample_{i:04d}.jpg')
        img.save(img_path)
        
        # Create sample caption
        caption = f"This is a sample image number {i} with random colors and patterns."
        captions.append(caption)
    
    # Save captions
    with open(os.path.join(data_dir, 'captions.txt'), 'w') as f:
        for i, caption in enumerate(captions):
            f.write(f'images/sample_{i:04d}.jpg\t{caption}\n')
    
    print(f"Created {num_samples} sample images and captions in {data_dir}")