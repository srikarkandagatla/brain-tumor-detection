# brain-tumor-detection
Brain tumor detection using CNN for multi-class classification.


The dataset comprises two main folders, "training" and "testing," encompassing 5712 MRI scan images of brains in the training set. These images are categorized into four classes: Glioma, Meningioma, No Tumor, and Pituitary. The testing set mirrors this structure but contains 1311 images.

You can access the dataset here: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/

Within this repository, you'll find two Python notebooks. In "Brain Tumor Detection.ipynb," a Convolutional Neural Network (CNN) model is implemented, with an input size of 150x150 pixels in RGB colors. The architecture comprises 14 layers for the CNN component and an additional 6 layers for the Artificial Neural Network (ANN) part. Augmentation techniques from the ImageDataGenerator class are applied to enhance model performance. The achieved validation accuracy is approximately 88.33%, demonstrating effectiveness during testing.

In the second notebook, "Brain Tumor Detection (VGG16).ipynb," a transfer learning approach is employed using the pre-trained VGG-16 model, originally trained on the ImageNet Dataset. The input size for this model is 224x224 pixels in all three color channels. Notably, the model attains a commendable validation accuracy score of 95.80% on testing images.

Both implementations incorporate a custom callback function designed to halt training when a specified accuracy score is achieved consecutively for a set number of times.

Note: CNN stands for Convolutional Neural Network, and VGG-16 represents the 16-layer VGGNet architecture.
