# Semantic Segmentation Model

## Overview

This repository contains the implementation of a semantic segmentation model using a slight variation of the U-Net architecture. The model is designed to perform image segmentation on the Labeled Faces in the Wild (LFW) dataset. It leverages PyTorch for model building and training, and integrates Weights & Biases (wandb) for experiment tracking and logging.

## Features

- **U-Net Architecture**: Custom implementation of the U-Net model with encoder and decoder blocks.
- **Dataset Handling**: Automated downloading and preprocessing of the LFW dataset.
- **Training and Validation**: Functions to train and validate the model, including logging of metrics and predictions.
- **Model Checkpointing**: Custom checkpointing system to save and log the model at various stages of training.
- **Wandb Integration**: Experiment tracking and logging with wandb, including configuration management and artifact logging.

## Requirements

- python 3.9
- torch
- torchvision
- wandb
- numpy
- matplotlib
- tqdm
- requests
- scikit-learn
- pillow
- gradio

## Installation

To install the required libraries, run:

```bash
pip install -r requirements.txt
```
## Usage

1. **Initialization**: Start a wandb project by setting up an account on [wandb.ai](https://wandb.ai).
2. **Dataset**: The `LFWDataset` class will automatically download and preprocess the LFW dataset.
3. **Training**: Run the main script to start training the model. Adjust parameters like learning rate, epochs, and batch size as needed.
4. **Monitoring**: Monitor the training progress and metrics on your wandb dashboard.

## File Structure

- `model/UNet.py`: Contains the U-Net model architecture.
- `dataset.py`: Defines the `LFWDataset` class for dataset handling.
- `eval_metrics.py`: Functions for calculating segmentation metrics.
- `log.py`: Utility functions for logging predictions.
- `model_checkpoint.py`: The `ModelCheckpoint` class for model checkpointing.
- Main script: Orchestrates the training and validation process.

## Configuration

Edit the main script to configure the following parameters:
- `learning_rate`: Learning rate for the optimizer.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for training and validation.

## Contributing

Contributions to improve the model or add new features are welcome. Please submit a pull request or open an issue for discussion.

## Contact

For any queries or suggestions, please open an issue in the repository.

---
