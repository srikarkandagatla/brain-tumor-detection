# brain-tumor-detection
Brain tumor detection using CNN for multi-class classification.

## About Dataset
The dataset comprises two main folders, "training" and "testing," encompassing 5712 MRI scan images of brains in the training set. These images are categorized into four classes: Glioma, Meningioma, No Tumor, and Pituitary. The testing set mirrors this structure but contains 1311 images.

You can access the dataset here: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/

## Repository Overview
This repository contains two folders named 0.85 and 0.95, which represent the accuracy thresholds at which the training of models has stopped using a custom stopping function.

## Accuracy (0.85)
- The 0.85 folder includes 5 Jupyter notebooks (.ipynb files) where two models are utilized:
 - Custom Model (BTD_GPU)
 - Pre-trained VGG16 Model

- Training Details:
 - Custom Model (BTD_GPU): Trained on three different input sizes (150x150, 192x192, 224x224).
 - VGG16 Model: Trained on two input sizes (192x192, 224x224).
All models in this folder were trained until they first achieved a validation accuracy of 0.85, using a custom callback function to halt training.

## Accuracy (0.95)
The 0.95 folder contains the same models as in the 0.85 folder, with identical input sizes and configurations. However, these models were allowed to train until they first achieved a validation accuracy of 0.95, using the same custom callback function.
Each file in both folders includes the time taken for training each model.
Both implementations incorporate a custom callback function designed to halt training when a specified accuracy score is achieved consecutively for a set number of times.
 - Note: CNN stands for Convolutional Neural Network, and VGG-16 represents the 16-layer VGGNet architecture.
