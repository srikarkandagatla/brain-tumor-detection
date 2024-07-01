# brain-tumor-detection
Brain tumor detection using CNN for multi-class classification.

## About Dataset
The dataset comprises two main folders, "training" and "testing," encompassing 5712 MRI scan images of brains in the training set. These images are categorized into four classes: Glioma, Meningioma, No Tumor, and Pituitary. The testing set mirrors this structure but contains 1311 images.

You can access the dataset [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/).

## Repository Overview
This repository contains four folders named Accuracy(0.85), Accuracy(0.95), 'Brain Tumor Detection (VGG16).ipynb', and 'Brain Tumor Detection.ipynb' which represent the accuracy thresholds at which the training of models has stopped using a custom stopping function.

## Accuracy (0.85)
**The '0.85' folder includes 5 Jupyter notebooks ('.ipynb' files) where two models are utilized**:
 - Custom Model (BTD_GPU)
 - Pre-trained VGG16 Model

**Training Details**:
 - Custom Model (BTD_GPU): Trained on three different input sizes (150x150, 192x192, 224x224).
 - VGG16 Model: Trained on two input sizes (192x192, 224x224).
All models in this folder were trained until they first achieved a validation accuracy of '0.85', using a custom callback function to halt training.

## Accuracy (0.95)
The 0.95 folder contains the same models as the 0.85 folder, with identical input sizes and configurations. However, these models were allowed to train until they first achieved a validation accuracy of '0.95', using the same custom callback function.
Each file in both folders includes the time taken for training each model.
Both implementations incorporate a custom callback function designed to halt training when a specified accuracy score is achieved consecutively for a set number of times.
 - Note: CNN stands for Convolutional Neural Network, and VGG-16 represents the 16-layer VGGNet architecture.

## Summary (0.85, 0.95)
| Model                   | Input Sizes               |
|-------------------------|---------------------------|
| Custom Model (BTD_GPU)  | 150x150, 192x192, 224x224 |
| Pre-trained VGG16 Model | 192x192, 224x224          |

**Table – 1:** All models in this folder were trained until they first achieved validation accuracies of '0.85' and '0.95', using a custom callback function to halt training.

## Brain Tumor Detection (VGG16).ipynb
In the second notebook, "Brain Tumor Detection (VGG16).ipynb," a transfer learning approach is employed using the pre-trained VGG-16 model, originally trained on the ImageNet Dataset. The input size for this model is 224x224 pixels in all three color channels. Notably, the model attains a commendable validation accuracy score of 95.80% on testing images.

## Brain Tumor Detection.ipynb
In "Brain Tumor Detection.ipynb," a Convolutional Neural Network (CNN) model is implemented, with an input size of 150x150 pixels in RGB colors. The Convolutional Neural Network (CNN) part consists of multiple convolutional layers followed by max-pooling layers. Augmentation techniques from the ImageDataGenerator class are applied to enhance model performance. The achieved validation accuracy is approximately 88.33%, demonstrating effectiveness during testing.

## Results
These values are expressed as ratios between the correctly predicted samples and the total number of samples in the validation set. These values range from 0 to 1, where:
- 0 represents 0% accuracy (all predictions are incorrect).
- 1 represents 100% accuracy (all predictions are correct).
The models "BTD" and "VGG_BTD" are allowed to be trained for a maximum of 1000 epochs, until they achieve a validation accuracy of '0.85' under the conditions described in table - 1.

### Training Details
**a) Custom Callback Function (0.85 ≤ val_accuracy)**
| Size       | Epochs | Accuracy | Loss   | Val Accuracy | Val Loss | Time(Sec) |
|:-----------|:------:|:--------:|:------:|:------------:|:--------:|:-------:|
| 150 x 150  | 29     | 0.8554   | 0.4116 | 0.8825       | 0.3209   | 1772    |
| 192 x 192  | 32     | 0.8477   | 0.4151 | 0.8535       | 0.4005   | 2379    |
| 224 x 224  | 35     | 0.8741   | 0.3572 | 0.8619       | 0.3549   | 3040    |
| VGG (192)  | 1      | 0.8318   | 0.4579 | 0.8734       | 0.2987   | 41      |
| VGG (224)  | 1      | 0.8349   | 0.4790 | 0.8886       | 0.3167   | 50      |

**Table – 2:** The training paused when the model achieved a validation accuracy of '0.85' or higher for the first time.

**b) Custom Callback Function (0.95 ≤ val_accuracy)**
| Size       | Epochs | Accuracy | Loss   | Val Accuracy | Val Loss | Time(Sec) |
|:-----------|:------:|:--------:|:------:|:------------:|:--------:|:---------:|
| 150 x 150  | 110    | 0.9412   | 0.1723 | 0.9527       | 0.1659   | 6593      |
| 192 x 192  | 107    | 0.9461   | 0.1619 | 0.95124      | 0.1394   | 7658      |
| 224 x 224  | 66     | 0.9335   | 0.1921 | 0.9565       | 0.1403   | 7469      |
| VGG (192)  | 6      | 0.9722   | 0.07921| 0.9626       | 0.1103   | 154       |
| VGG (224)  | 5      | 0.9632   | 0.1024 | 0.9637       | 0.0930   | 212       |

**Table – 3:** The training paused when the model achieved a validation accuracy of '0.95' or higher for the first time.
