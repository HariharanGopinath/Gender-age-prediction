# Gender-age-prediction-using the deep learning model

In this project, I will discuss an interesting application of Deep Learning applied to faces. I will estimate the age and figure out the gender from a single image or from live video detection.

I have divided this project into 2 models. In the model 1, I have predicted the age and gender of a person through a object detection method either through a live camera or a photo. In the model 2, I have performed a prediction model through a general deep learning model where we can predict only gender with photos.

# Model 1
The model is trained by Gil Levi and Tal Hassner (https://talhassner.github.io/home/publication/2015_CVPR). 

## 1. Gender and Age Classification using CNNs
The authors have used a very simple convolutional neural network architecture, similar to the CaffeNet and AlexNet. The network uses 3 convolutional layers, 2 fully connected layers and a final output layer. The details of the layers are given below.

Conv1 : The first convolutional layer has 96 nodes of kernel size 7.
Conv2 : The second conv layer has 256 nodes with kernel size 5.
Conv3 : The third conv layer has 384 nodes with kernel size 3.
The two fully connected layers have 512 nodes each.

They have used the Adience dataset for training the model (https://talhassner.github.io/home/projects/Adience/Adience-data.html).

### 1.1. Gender Prediction
They have framed Gender Prediction as a classification problem. The output layer in the gender prediction network is of type softmax with 2 nodes indicating the two classes “Male” and “Female”.

### 1.2. Age Prediction
Ideally, Age Prediction should be approached as a Regression problem since we are expecting a real number as the output. However, estimating age accurately using regression is challenging. Even humans cannot accurately predict the age based on looking at a person. However, we have an idea of whether they are in their 20s or in their 30s. Because of this reason, it is wise to frame this problem as a classification problem where we try to estimate the age group the person is in. For example, age in the range of 0-2 is a single class, 4-6 is another class and so on.

The Adience dataset has 8 classes divided into the following age groups [(0 – 2), (4 – 6), (8 – 12), (15 – 20), (21 – 32), (38 – 43), (48 – 53), (60 – 100)]. Thus, the age prediction network has 8 nodes in the final softmax layer indicating the mentioned age ranges.

# Code details

The code can be divided into four parts:

    1) Detect Faces
    2) Detect Gender
    3) Detect Age
    4) Display output
    
NOTE: Please download the model weights file
Gender-https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0
Age-https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0
    
    
# Results
My face
![result1](https://user-images.githubusercontent.com/71879067/127671477-717d5d18-67e4-448b-99ad-6f7c1fb06970.JPG)
![result2](https://user-images.githubusercontent.com/71879067/127671503-d5764b0e-92dd-4b30-ac92-65e41803c46e.JPG)

We saw above that the network is able to predict both Gender and Age to high level of accuracy.

# Model-2

Gender Classification with Python\ Machine Learning Project 

In this project, I’ll walk you through a machine learning project on gender classification with the Python programming language.

Dataset can be downloaded from kaggle website through the below link 
 https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba/download

This dataset consist of thousands of images for the male and female for training, validating and testing our machine learning model

In this project, I will use the some python libraries like numpy, tensorflow and matplotlib for building our machine learning model.

In the final result, I got almost 80% of training and validation accuracy with less than 0.5% losses and successful in predicting the gender in the photo.


