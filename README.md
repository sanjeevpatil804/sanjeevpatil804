# StackGAN Text-to-Image Generation

A comprehensive implementation of StackGAN for text-to-image generation using PyTorch. This project includes a complete training pipeline, inference capabilities, and interactive demos.

## About Me
I'm a recent graduate with master's degree in data science, with a passion for unlocking insights from data and driving decision-making through machine learning and data analysis. My journey in data science began during my academic years, where I honed my skills in Python, R, and SQL, and developed a strong foundation in statistics and machine learning.

## Skills
- **Programming Languages:** Python, SQL
- **Data Analysis Tools:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning Libraries:** Scikit-learn, NLTK, Tensorflow/Keras, Pytorch, OpenCV
- **Specializations:** Computer Vision, Natural Language Processing, Time Series Analysis, Generative AI
- **Data Management:** SQL, PostgreSQL

## Projects
During my course, I worked on various projects which are showcased in my repositories. Here are a few highlights:
- **Generative AI - StackGAN:** Utilized StackGan for text to image generation using CUB dataset. Developed WGAN model for face generation using CelebA dataset.
- **Predictive Modeling for House price:** Utilized regression models to forecast saling price, achieving an 85% accuracy.
- **Sentiment Analysis on Social Media Text:** Developed a NLP model to analyze customer sentiments on Twitter data.
- **Data Visualization Dashboard:** Created an interactive dashboard using Tableau to visualize bitcoin trends, Estcy shops trends etc.

## StackGAN Project Overview

### Architecture
StackGAN is a two-stage generative adversarial network that generates high-resolution images from text descriptions:

1. **Stage-I Generator**: Generates 64×64 images from text descriptions
2. **Stage-II Generator**: Generates 256×256 images from Stage-I outputs and text descriptions
3. **Text Encoder**: LSTM-based encoder for processing text descriptions
4. **Discriminators**: Conditional discriminators for both stages

### Key Features
- **Two-stage generation**: Coarse-to-fine image generation
- **Conditional GAN**: Text-conditioned image generation
- **Gradient penalty**: WGAN-GP for training stability
- **Feature matching**: Improved training with feature matching loss
- **Comprehensive logging**: TensorBoard integration for monitoring
- **Interactive demo**: Real-time text-to-image generation

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd stackgan-text-to-image

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for text processing)
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Quick Demo
Run the complete demo to see StackGAN in action:
```bash
python demo.py --mode full
```

This will:
1. Create sample data
2. Train a model for a few epochs
3. Generate images from text descriptions
4. Start interactive mode

### Training

#### With Sample Data
```bash
# Create sample data and train
python train.py --data_dir data/sample --create_sample --num_epochs 50
```

#### With Custom Dataset
Prepare your dataset in the following format:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt
```

Format of `captions.txt`:
```
images/image1.jpg    A beautiful red flower in a garden
images/image2.jpg    A majestic mountain landscape at sunset
```

Then train:
```bash
python train.py --data_dir data/your_dataset --num_epochs 100 --batch_size 16
```

#### Training Options
```bash
python train.py \
    --data_dir data/cub \
    --batch_size 16 \
    --num_epochs 100 \
    --lr_g 0.0002 \
    --lr_d 0.0002 \
    --device cuda \
    --log_dir logs \
    --checkpoint_dir checkpoints
```

### Inference

#### Command Line
```bash
python inference/generator.py \
    --model_path checkpoints/stackgan_epoch_100.pth \
    --vocab_path data/vocabulary.pkl \
    --text "A beautiful sunset over the ocean" \
    --num_images 4 \
    --output_dir generated_images
```

#### Interactive Mode
```bash
python inference/generator.py \
    --model_path checkpoints/stackgan_epoch_100.pth \
    --vocab_path data/vocabulary.pkl \
    --interactive
```

### Demo Modes

```bash
# Create sample data only
python demo.py --mode data

# Train model only
python demo.py --mode train --num_epochs 10

# Generate images only (requires trained model)
python demo.py --mode generate

# Interactive demo only
python demo.py --mode interactive

# Full demo (recommended)
python demo.py --mode full
```

## Project Structure

```
stackgan-text-to-image/
├── models/
│   └── stackgan.py          # StackGAN model implementation
├── data/
│   └── dataset.py           # Dataset classes and data loading
├── training/
│   └── trainer.py           # Training loop and loss functions
├── inference/
│   └── generator.py         # Inference and image generation
├── train.py                 # Main training script
├── demo.py                  # Demo script
├── requirements.txt          # Dependencies
└── README.md               # This file
```

## Model Architecture

### Stage-I Generator
- Input: Text features + Noise (100-dim)
- Output: 64×64 images
- Architecture: Transposed convolutions with batch normalization

### Stage-II Generator
- Input: Stage-I images + Text features + Noise
- Output: 256×256 images
- Architecture: Transposed convolutions with residual connections

### Text Encoder
- Input: Text tokens
- Architecture: Bidirectional LSTM
- Output: Text features (512-dim)

### Discriminators
- Conditional discriminators for both stages
- Architecture: Convolutional layers with text conditioning
- Loss: WGAN-GP with gradient penalty

## Training Details

### Loss Functions
- **Adversarial Loss**: WGAN-GP for training stability
- **Feature Matching Loss**: L1 loss between real and fake features
- **Gradient Penalty**: Ensures Lipschitz constraint

### Hyperparameters
- Learning rate: 0.0002 (both generator and discriminator)
- Batch size: 16
- Noise dimension: 100
- Text embedding dimension: 256
- Gradient penalty weight: 10.0

### Monitoring
- TensorBoard logging for loss curves
- Sample generation every N epochs
- Checkpoint saving
- Validation metrics

## Results

The model generates images in two stages:
1. **Stage-I**: 64×64 low-resolution images with basic structure
2. **Stage-II**: 256×256 high-resolution images with fine details

Sample outputs show the progression from text description to high-quality images.

## Customization

### Model Architecture
Modify `models/stackgan.py` to change:
- Network architectures
- Layer dimensions
- Activation functions

### Training Parameters
Adjust in `train.py`:
- Learning rates
- Batch sizes
- Number of epochs
- Loss weights

### Data Format
Support for custom datasets by implementing:
- Custom dataset classes
- Text preprocessing
- Image transformations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Use CPU: `--device cpu`

2. **Training Instability**
   - Adjust learning rates
   - Increase gradient penalty weight
   - Use smaller model dimensions

3. **Poor Image Quality**
   - Train for more epochs
   - Increase model capacity
   - Improve data quality

### Performance Tips

1. **GPU Usage**
   - Use CUDA for faster training
   - Monitor GPU memory usage
   - Adjust batch size accordingly

2. **Data Loading**
   - Use multiple workers: `--num_workers 4`
   - Pin memory for faster data transfer
   - Preprocess data if possible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Connect with Me
- LinkedIn: www.linkedin.com/in/sanjeev-patil-706a27230/
- Email: sanjeevpatil804@aol.com

Feel free to browse through my projects and reach out if you have any questions or collaboration ideas!
