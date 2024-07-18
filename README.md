# CGAN-Project
This repository contains the code for a Conditional Generative Adversarial Network (cGAN) project, which generates images conditioned on specific labels. This project involves several steps, including image preprocessing, data augmentation, model building, training, and saving the generated images and labels.

# Table of Contents
- Project Overview
- Dataset
- Dependencies
- Usage
- Model Architecture
- Training
- Results
  
## Project Overview
This project implements a cGAN to generate images conditioned on specific labels. The main components include:

- Image Preprocessing: Resize and convert images to grayscale.
- Data Augmentation: Generate additional images through transformations like rotation and flipping.
- Model Building: Construct the generator and discriminator models using Keras.
- Training: Train the cGAN using a combination of real and generated images.
- Saving Results: Save generated images and their corresponding labels.

## Dataset
The dataset used in this project consists of images stored in a local directory. The images are preprocessed to 28x28 pixels and converted to grayscale before being fed into the cGAN.

## Dependencies
To run this project, you need the following dependencies:
- Python 3.x
- NumPy
- Pandas
- Keras
- TensorFlow
- PIL (Pillow)
- Matplotlib
- SciPy

## Usage
- Clone the repository:
- Set up your dataset:
- Run the preprocessing and augmentation scripts:
- Train the cGAN:
- Generate images:

## Model Architecture
### Generator
The generator model takes random noise and a label as input and produces an image. It uses layers such as Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, and LeakyReLU.

### Discriminator
The discriminator model takes an image and a label as input and predicts whether the image is real or fake. It uses layers such as Conv2D, BatchNormalization, LeakyReLU, Dropout, Flatten, and Dense.

### Training
The cGAN is trained using the following steps:

- Real and Fake Labels: Generate real and fake labels for training.
- Train Discriminator: Train the discriminator on real and generated images.
- Train Generator: Train the generator to produce images that can fool the discriminator.

### Results
The trained cGAN generates images conditioned on specific labels. Sample generated images and their corresponding labels are saved in the generated_images directory.
