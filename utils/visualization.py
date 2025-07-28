import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os


def save_image_grid(images, save_path, nrow=8, padding=2, normalize=True):
    """Save a grid of images"""
    if normalize:
        images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    
    # Convert to numpy and transpose
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Create grid
    ncol = len(images) // nrow
    if len(images) % nrow != 0:
        ncol += 1
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        if row < nrow:
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), nrow * ncol):
        row = i // ncol
        col = i % ncol
        if row < nrow:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_image(real_images, fake_images, save_path, title="Real vs Generated"):
    """Create a comparison image between real and generated images"""
    num_images = min(len(real_images), len(fake_images))
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_images):
        # Real image
        real_img = real_images[i].cpu().permute(1, 2, 0).numpy()
        real_img = (real_img + 1) / 2
        axes[0, i].imshow(real_img)
        axes[0, i].set_title(f'Real {i+1}')
        axes[0, i].axis('off')
        
        # Generated image
        fake_img = fake_images[i].cpu().permute(1, 2, 0).numpy()
        fake_img = (fake_img + 1) / 2
        axes[1, i].imshow(fake_img)
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(losses, save_path):
    """Plot training loss curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Stage-I losses
    if 'stage1_d_loss' in losses:
        axes[0, 0].plot(losses['stage1_d_loss'])
        axes[0, 0].set_title('Stage-I Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
    
    if 'stage1_g_loss' in losses:
        axes[0, 1].plot(losses['stage1_g_loss'])
        axes[0, 1].set_title('Stage-I Generator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
    
    # Stage-II losses
    if 'stage2_d_loss' in losses:
        axes[1, 0].plot(losses['stage2_d_loss'])
        axes[1, 0].set_title('Stage-II Discriminator Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
    
    if 'stage2_g_loss' in losses:
        axes[1, 1].plot(losses['stage2_g_loss'])
        axes[1, 1].set_title('Stage-II Generator Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    image = tensor.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)


def save_tensor_as_image(tensor, save_path):
    """Save tensor as image file"""
    pil_image = tensor_to_pil(tensor)
    pil_image.save(save_path)


def create_progress_animation(images_list, save_path, fps=2):
    """Create an animation showing the progression of generated images"""
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def animate(frame):
        ax.clear()
        img = images_list[frame].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        ax.imshow(img)
        ax.set_title(f'Frame {frame + 1}')
        ax.axis('off')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(images_list), interval=1000//fps)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()


def visualize_text_embeddings(embeddings, texts, save_path):
    """Visualize text embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add text labels
    for i, text in enumerate(texts):
        plt.annotate(text[:20] + '...' if len(text) > 20 else text, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.8)
    
    plt.title('Text Embeddings Visualization (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()