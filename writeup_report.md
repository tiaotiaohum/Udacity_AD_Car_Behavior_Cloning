# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/loss_vari_5epochs.png "RMSE sequences"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 59-73).

The model includes RELU layers to introduce nonlinearity (code line 67, 68), and the data is normalized in the model using a Keras lambda layer (code line 63).

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Because the left turns dominante the szenarios. So I also drove the car in another direction to generate more data with right turn, so that the model would be more general valid.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict the steering angles with the input road images.

I planned to use several convolutional layers and full connected lay to achive this goal.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used mean squared error to measure the performance of the model, applied to the validation dataset.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Attempts to reduce overfitting in the model

About the overfitting, I monitorred the RMSE error values along more epochs, unsed the "tuning the epochs number" to avoid overfitting.

A visualisation is shown below.
![alt text][image8]

In the diagram, I saw after 2nd epochs the accuracy on the train data reduces, while the accuracy on the validation data increases. This is clearly a overfitting.

So the training epochs is set to 2 to avoid overfitting (model.py lines 80). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. So the dropout layer is not neccessary in this situation.

#### 3. Final Model Architecture

The final model architecture (model.py lines 62-73) consisted of a convolution neural network with the following layers as written below.

A over view of each layers is following:
* Cropping the picture for focusing on the road
* Normalization
* 2D convolutional layer with depth: 24, filter size: 5 * 5
* 2D convolutional layer with depth: 36, filter size: 5 * 5
* 2D convolutional layer with depth: 48, filter size: 5 * 5
* 2D convolutional layer with depth: 64, filter size: 3 * 3
* 2D convolutional layer with depth: 64, filter size: 3 * 3
* Flatten
* Full connected layer with the output size 100
* Full connected layer with the output size 50
* Full connected layer with the output size 10
* Full connected layer with the output size 1

#### 4. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process towards another driving direction on the same track two in order to get more data points and redce the training data bias to left turn.

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
