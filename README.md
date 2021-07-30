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


