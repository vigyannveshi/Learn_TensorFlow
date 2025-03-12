# Learn Tensorflow

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/2560px-TensorFlow_logo.svg.png" width="250">
</p>

The repository is created to share content that I studied and resources I have used to learn Tensorflow.

### Disclaimer: 

I am studying this content refering to course: `TensorFlow for Deep Learning Bootcamp` by `Andrei Neagoie` and `Daniel Bourke` on Udemy: 

https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/ 

I shall be refering the content, and add my understandings and works to this repository and not the actual course work.

The creators of the course have worked very hard in their content, and please refer their course for their original work, which is far more than what I intend to reflect upon. 


### Contents:

| <u>**1. Tensorflow Fundamentals**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
| --> Creating tensors using `tf.constant()` | 1 |
| --> Creating tensors using `tf.Variable()` | 1 |
| --> Creating random tensors using `tf.random.distribution()` | 1 |
| --> Shuffle the order of elements of tensor using `tf.random.shuffle()` | 2 |
| --> Other ways to create tensors | 3 |
| --> Getting information from tensors | 4 |
| --> Indexing and expanding tensors | 5 |
| --> Reshaping the tensor using `tf.reshape()` | 5 |
| --> Adding an extra dimension to tensor using `tf.newaxis` or `tf.expand_dims()` | 5 |
| --> Manipulating tensors with basic operations (+,-,*,/) | 6 |
| --> Matrix Multiplication | 7 |
| --> Dot Product | 7 |
| --> Changing datatype of tensors using `tf.cast()` | 8 |
| --> Aggregating tensors | 9 |
| --> Finding positional maximum and minimum | 10 |
| --> Squeezing a tensor `tf.squeeze()` | 11 |
| --> One Hot Encoding `tf.one_hot(indices:tensor,depth:"number of classes")` | 12 |
| --> Some more tensor functions and operations | 13 |
| --> Tensors and Numpy | 14 |
| --> Understanding Devices to speed up TensorFlow | 15 |
| --> Exercise | 16 |

<br>
<br>


| <u>**2. Neural Network Regression with Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Creating data for regression model| 1 |
|--> Understanding input and output shapes (features and labels) | 1 |
|--> Major steps in modelling with tensorflow | 2 |
|--> Improving the model | 3 |
|--> Evaluating the model | 4 | 
|--> Train-Validate-Test split | 4 | 
|--> Visualizing the data | 4 | 
|--> Visualizing the model | 4 |  
|--> Visualizing model predictions | 5 |  
|--> Evaluating model's prediction in regression| 5 |  
|--> Saving and loading a model | 5 |
|--> Capstone Project 1|main_1, main_2|
|--> Capstone Project 2|main_1|

| <u>**3. Customizing Models**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Custom training loop |1|
|--> Custom loss function |2|
|--> Custom model         |3|
|--> Improving Custom model |4|


| <u>**4. Neural Network Classification with Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Steps in modelling classification model with tensorflow |1|
|--> Using history of `model.fit()` |1|
|--> Finding the best learning rate |2|
|--> Classification Evaluation Methods |3|
|--> Multiclass classification         |4|
|--> Improving Predictions on Fashion MNIST using normalized data |5|
|--> Finding ideal learning rate for Fashion MNIST |6|
|--> Evaluating trained model on Fashion MNIST |6|
|--> Understanding patterns learnt by our model|7|


| <u>**5. Convolutional Neural Network & Computer Vision with Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Building the first CNN (baseline model)|1|
|--> Overcoming overfitting |2|
|--> Data augmentation |2|
|--> Making a prediction with our trained model on custom data |2|
|--> Multiclass image classification |3|
|--> Visualizing each layer and activation output |3|
|--> Adjust model hyperparams to reduce overfitting / beat the baseline model |4,5,6,7|
|--> Making a prediction with our trained model on custom data (multi-class) |7|
|--> Saving and loading our multi-class model |7|

| <u>**6. Transfer learning - feature extraction**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Setting up callbacks |1|
|--> Creating models using Tensorflow Hub |1|
|--> Building and Compiling Tensorflow feature extraction model |1|
|--> Resnet-50 v2 vs Efficient net b0 |1|
|--> Visualization using TensorBoard |1|