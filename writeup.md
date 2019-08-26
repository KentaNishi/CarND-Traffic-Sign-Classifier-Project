# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples.png "Visualization"
[image2]: ./prediction_test_result.png "Prediction result"
[image3]: ./examples_change_luminance.png "Change luminance"
[image4]: ./examples_change_perspective.png "Change perspective"
[image5]: ./figure_epoch200_rate_0.0001_batch128.png "Lenet result"
[image6]: ./self_difined_CNN_last_ver_with_data_augumentation_figure_epoch1000_rate_0.0001_batch1024_is_rgb1.png "Self difined CNN result"
[image7]: ./data_distribution.png "data distribution"
[image8]: ./new_images.png "images from web"
[image9]: ./data_distribution_with_data_augumentation.png "data distribution with data augumentation"
[image10]: ./prediction_test_result_with_data_augumentation.png "Prediction result with data augumentation"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/KentaNishi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many the data is included in training ,validation and test data for each type(label).

![alt text][image1]
![alt text][image7]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because to advance learning properly. For example, data1:(0~255),data2(0~1) it does not affect evenly,it is going to be data1 biased learning. So I normalized the data.

I decided to generate additional data to make my model learn more generally. 

To add more data to the the data set, I used the following techniques because 

1. change luminance
1. perspective transform
1. flip image
1. zoom

I used first method because I think driving environment has various luminance.
Last one is used because of a reason like above.
I use 2nd method to adapt change in viewing angle. When a viewing angle chages, a form in camera cordination changes.
In driving situation, I think that possible rotational direction are roll,pitch and yaw , because there are curve,slope and tilted roads.(x-y plane is same as image plane.x-axis:vertical,y-axis:horizontal)
I assume z value is always equal zero.
Third one is based on https://navoshta.com/traffic-signs-classification/


Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image4]

The difference between the original data set and the augmented data set is the following ... 


Before:

Number of training examples = 34799

Number of validation examples = 4410

Number of testing examples = 12630

![alt text][image7]

After:

Number of training examples = 439157

Number of validation examples = 55680

Number of testing examples = 159390

![alt text][image9]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 36x36x3 RGB image.(Images are padded in pre process) 	| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 32x32x16 	        |
| RELU					|												        |
| Batch Normalization   |                                                       |
| Max pooling	      	| 2x2 stride,  outputs 16x16x16			    	        |
| Drop out              | keep_prob = 0.8                                       |
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 12x12x32 	        |
| RELU					|												        |
| Batch Normalization   |                                                       |
| Convolution 5x5	    |  1x1 stride, Valid padding, outputs 8x8x80	        |
| Max pooling	      	| 2x2 stride,  outputs 4x4x80			    	        |
| Fully connected		| inputs:1280, outputs 300       				        |
| RELU					|												        |
| Fully connected		| inputs:300, outputs 100       				        |
| RELU					|												        |
| Fully connected		| inputs:100, outputs 43           				        |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

To train the model, I used an `AdamOptimizer`. In addition, I introduced early stopping algorythm to prevent the model from overfitting.

Hyperparameters are:

| Name            | Value  | Description                                   |
|:---------------:|:------:|:---------------------------------------------:|
| `mu`            | 0      | initilazing Wights with normal distribution   |
| `sigma`         | 0.1    | initilazing Wights with normal distribution   |
| `learning_rate` | 0.001  |                                               |
| `epochs`        | 1000   | It often stop 100~300                         |
| `BATCH_SIZE`    | 128    | without data augumentation                    |
| `BATCH_SIZE`    | 1024   | with data augumentation                       |

Early stopping algorythm is as follows:

```python
### accuracy array hold validation accuracy at each epoch.

if i>100:
            if ((accuracy_array[i-1]-np.max(accuracy_array)) < -0.01):
                break
            elif ((np.max(accuracy_array[i-21:i-1])-np.max(accuracy_array[i-41:i-21])) <= 0.0):
                break
            #endif
        #endif
```


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
INFO:tensorflow:Restoring parameters from ./self_difined_CNN_last_ver_with_data_augumentation_is_rgb1
* Train data Accuracy : 0.9978299332487207
* Train data Loss : 0.006321019644243348
* Valid data Accuracy : 0.9508261492882651
* Valid data Loss : 0.4036143529004064
* Test data Accuracy : 0.9327749545514804
* Test data Loss : 0.5342465304477249

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 My first architecture is Lenet. Because this was explaned in the class.
* What were some problems with the initial architecture?
 Its accuracy reaches less than 90%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  I added one more convolutional layer,because I think initial one's problem is shortage of expressing ability.
  In CNN,it is said that the first layer learns simple feature and deeper layer express more complicated feature and third one is often express a shape like objects you see in your daily lives.
  Besides,to avoid over fitting, I use Batch normalization and dropout. 
* Which parameters were tuned? How were they adjusted and why?
  Channel sizes and dropout's keep_probs were tuned. Former one is besed on their ability of expression. I judged based on accuracy. Latter is based on which the model is over fitting.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  I've explained above. 

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8]

The last image might be difficult to classify because it image is shifted and there is excess sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Without data augumentation.
![alt text][image2]

With data augumentation.
![alt text][image10]

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.71428571428571%. This compares favorably to the accuracy on the test set of 94.4%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a "Right-of-way at the next intersection" (probability of almost 1.0), and the image does contain a "Right-of-way at the next intersection". 
Only fifth image was mistaken, but the top five soft max probabilities contains correct answer.

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


