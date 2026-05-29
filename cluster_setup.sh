#!/bin/bash
# BGU Cluster Setup Script for CUT Training
# Run this script once after connecting to the GPU node

echo "=========================================="
echo "Setting up CUT Training Environment"
echo "=========================================="

# Create project directory
PROJECT_DIR=~/chess_cut_project
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Clone CUT repository
if [ ! -d "contrastive-unpaired-translation" ]; then
    echo "Cloning CUT repository..."
    git clone https://github.com/taesungp/contrastive-unpaired-translation.git
else
    echo "CUT repository already exists"
fi

cd contrastive-unpaired-translation

# Create conda environment (if conda is available)
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda create -n cut_env python=3.8 -y
    conda activate cut_env
fi

# Install PyTorch (for GTX 1080 Ti - CUDA 11.x compatible)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install dominate visdom packaging

# Verify GPU
echo ""
echo "Verifying GPU setup..."
nvidia-smi
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your dataset to: $PROJECT_DIR/contrastive-unpaired-translation/datasets/chess/"
echo "2. Run training with: bash train_cut.sh"
