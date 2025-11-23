# CVPDL 2025 Homework 3: Image Generation

## Overview
This project implements image generation for MNIST images using a custom model train from scratch. 

## Features
- Custom dataset handling for MNIST images and annotations.

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
- **Note: please put dataset `train` and `test` under the `src` directory**
- **Note: please run the scripts in the `src` directory**
  ```bash
  cd src
  ```

1. **Data Preparation**:
   - Run the data preparation script:
     ```bash
     python yolo_dataset.py
     ```
   - (Optional) Run the tuning script:
     ```bash
     python yolo_tune.py
     ```

2. **Training**:
   - Run the training script:
     ```bash
     python yolo_train.py
     ```
   - This will train the model, perform validation, and save checkpoints in `runs/detect/<model_name>` directory.
   - It will also save train and validation loss in `runs/detect/<model_name>/results.csv` directory.

3. **Inference**:
   - Use the inference script to run predictions on new images:
     ```bash
     python yolo_inference.py
     ```

4. **Evaluation: FID**:
   - FID with the training dataset
     ```bash
     python -m pytorch_fid path/to/images path/to/mnist
     ```
   - FID with the test dataset
     ```bash
     python -m pytorch_fid path/to/images path/to/mnist.npz
     ```
        - `path/to/images`: the folder of the generated images.
        - `path/to/mnist`: the folder of the training data.
        - `path/to/mnist.npz`: the precalculated mean and covariance of training data.

## Contributing
This is a homework project for NTU_CVPDL_2025. For questions, refer to the course materials or contact the instructor.
