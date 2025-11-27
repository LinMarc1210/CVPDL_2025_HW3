# CVPDL 2025 Homework 3: Image Generation

## Overview
This project implements image generation for MNIST images using a DDPM train from scratch. 

## Features
- DDPM implementation for mnist dataset
- FID calculation for result

## Installation

Set up the environment (please install `uv` first):
   - Using uv:
     ```bash
     uv sync
     ```
   - activate your environment
     ```bash
     .venv\Scripts\Activate.ps1 
     ```

   Required packages include:
   - PyTorch
   - Torchvision
   - pytorch-fid
   - Other dependencies as listed in `requirements.txt`

## Usage
- **Note: please add another folder `images` under `mnist` dataset, and put all 60000 training images under `mnist/images/`**
- **Note: please run the scripts in the `src` directory**
  ```bash
  cd src
  ```

1. **Training**:
   - Run the training script:
     ```bash
     python train.py
     ```
   - This will train the model, perform validation, and save model as `ddpm_mnist_custom.pth`

2. **Inference**:
   - Use the inference script to run predictions on new images:
     ```bash
     python inference.py
     ```
   - This will generate 10000 images in the path `generated_images_10k/`

3. **Evaluation: FID**:
   - FID with the training dataset
     ```bash
     python -m pytorch_fid generated_images_10k mnist.npz
     ```
   - the result FID: `29.965259357164655`

4. **Visualization**:
   - Draw diffusion process (8x8)
     ```bash
     python visual.py
     ```
   - the result diffusion processs will be saved as `diffusion_process_8x8.png`


## Contributing
This is a homework project for NTU_CVPDL_2025. For questions, refer to the course materials or contact the instructor.
