# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/first_run_bad.jpg "First Run Image Bad"
[image2]: ./writeup_imgs/first_run_bad_flipped.jpg "Flipped Image"
[image3]: ./writeup_imgs/backwards_run.jpg "Backwards Run"
[image4]: ./writeup_imgs/center_example.jpg "Center Run"
[image4]: ./writeup_imgs/right_example.jpg "Left Run"
[image4]: ./writeup_imgs/left_example.jpg "Right Run"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results and explaining the approach taken to train the car.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is almost the same as the one used on the Nvidia blog post about a self driving car architecture.
It consists of 3 convolutional layers with a 5x5 filter size, 2x2 stride and 24, 36 and 48 filters each.
Followed, by two convolutional layers with 3x3 filter size, 1x1 stride size and 64 filters.
All of the convolutional layers use the `relu` activation function in order to introduce non linearity.
The last two convolutional layers have dropout layers with a probability of 0.5 in order to regularized the weights and avoid overfitting.

Finally the model has 4 fully connected layers with 1164, 200, 50, 10 units. The first fully connected layer is also followed
by a dropout layer with a 0.5 probability.

#### 2. Attempts to reduce overfitting in the model

As it was mentioned before, to prevent the model from overfitting the model has three different dropout layers.
Two of them are located in the last Convolutional layers and the third one located right after the first Dense layer.

In addition, the model was tested frequently to ensure that it was not overfitting the testing data.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Data was so important on this project. I collected a full lap and another lap one in reverse worth of data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall model development strategy  was to use something similar to what worked on the Traffic Sign Classifier project.
I started by using a similar architecture by adding multiple Convolutional layers, followed by max pooling layers and finally a series
or dense layers. During this step in the development, I used a single track lap of data and only the center images.
This solutioned worked well but my model loss stayed at 0.07.
After this, I tried to use the NVIDIA self driving car network architecture. It was interesting to see the importance of increasing field receptive to gather as much feature information from the image as possible by adding a series of convolutional layers without pooling layers.

This solution gave me better results, and I experimented by removing the last convolutional layer and adding an Inception Module. This only made the
model slower and did not improve accuracy so I decided to remove it.

I also tried to turn the images to grayscale in order to reduce the validation loss, however I realized that this would not work because the model is tested with
color images and more work needed to be done in order to add this functionality.

Up to this point, I focused on reduced the loss and I added three dropout layers in order to reduce overfitting since the validation loss was getting bigger
while the training loss increased.
When I tested the model it performed badly on the sharp turns, so I decided to add more data.
I added enhanced images by flipping the center images my first dataset and added another set of data by recording a lap backwards.
This improve the model but it was still not good enough and it drifted from the center in the sharp corners.

During this time, I realized that acquiring more data was key to improve the model testing performance even though this caused the loss to increased by a little bit.
I re recorded the data and made sure to stay in the center of the road by driving the car slowly. In addition, I decided to add the left and right images and use a correction factor of +/-0.2.
My final testing dataset consisted of the two laps of data, one forward the another one backwards, with the center, left, right raw and flipped images.

This dataset was split in the testing set(80%) and a validation set(20%).

Just by adding more data the model was finally able to stay in the center even in the hard parts of the road such as the sharp corners, the bridge and the section
with a missing lane.

#### 2. Final Model Architecture

Here is the a summary of the final model architecture.
Model Summary:
Layer (type)                     Output Shape          Param #     Connected to                    
====================================================================================================
input_2 (InputLayer)             (None, 160, 320, 3)   0                                           
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 160, 320, 3)   0           input_2[0][0]                   
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 65, 320, 3)    0           lambda_2[0][0]                  
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_2[0][0]              
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_6[0][0]           
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_7[0][0]           
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_8[0][0]           
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 3, 35, 64)     0           convolution2d_9[0][0]           
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 1, 33, 64)     36928       dropout_4[0][0]                 
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1, 33, 64)     0           convolution2d_10[0][0]          
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 2112)          0           dropout_5[0][0]                 
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1164)          2459532     flatten_2[0][0]                 
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 1164)          0           dense_6[0][0]                   
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 200)           233000      dropout_6[0][0]                 
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 50)            10050       dense_7[0][0]                   
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 10)            510         dense_8[0][0]                   
____________________________________________________________________________________________________
dense_10 (Dense)                 (None, 1)             11          dense_9[0][0]                   
====================================================================================================
Total params: 2,834,451
Trainable params: 2,834,451
Non-trainable params: 0
____________________________________________________________________________________________________
None
dict_keys(['loss', 'val_loss'])

#### 3. Creation of the Training Set & Training Process

At the beginning I drove the car at max speeds because I thought that speed was one of the models results. The car was easy to handle on the straight
parts of the road, but when drove around the sharp turns I ended touching side lanes. I thought that this was not important and the the model will learn how to
stay straight.
I also drove the car backwards to add more training data.

Here is an example of a sharp turn image:

![alt text][image1]

The validation loss was not decreasing at the rates that I expected to I enhanced the dataset by adding flipping the images in the whole dataset.
Here is an example of a flipped image:

![alt text][image2]

When I realized that data quality is important for the testing performance, I decided to re record the other lap but this time driving slow in order to keep the car in the center as much as possible. Also, I took advantage of the left and right images.

Here are some examples of the center, left and right images:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Also, for the backwards run lap:

![alt text][image3]



Finally, when I used the generator I decided to get all of the csv data(forward, backward laps and flipped versions of the laps)  in one variable and then shuffle it before splitting it into the testing and validation set.
The added shuffling to the model by using the `shuffle` parameter in the keras fit method and used the adam optimizer to avoid running a lot of hyper parameters.

I learned that data, data and more data are the most important factors to improve testing performance of a model. Even though, fancy neural networks architectures might help on reducing loos, it was the data what made the car stay in the center.



