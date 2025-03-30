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

| <u>**7. Transfer learning - fine tuning**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Using `tf.keras.preprocessing.image_dataset_from_directory()` |1|
|--> Performing Various Experimentations |1|
|--> --> Building a baseline model for running experiments |1|
|--> --> --> Building model using `Functional()` API |1|
|--> --> Getting feature vector from trained model |2|
|--> --> Running a model experiment using 1% of training data|3|
|--> --> Running a model experiment using 10% of augmented training data|4|
|--> --> --> Creating a model checkpoint callback|4|
|--> --> --> Loading in model/model weights using saved checkpoints|4|
|--> --> Improving the baseline model, by fine tuning weights of base model |5|
|--> --> --> Loading the checkpoints of model 2 |5|
|--> --> --> Fine tuning for another 5 epochs |5|
|--> --> --> Comparing before and after fine-tuning |5|
|--> --> Running a model experiment using 100% of augmented training data with fine tuning|6|

| <u>**8. Transfer learning - scaling up**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|-->Training a model to fit data of 101 classes of Food101 dataset|1|
|-->Evaluating the fitted model|1|
|-->Visualizing f1-scores for different classes|1|
|-->Checking out where our model is most wrong|1|
|-->Prediction on custom images|1|


| <u>**9. Capstone - Project: Food Vision**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Using Tensorflow datasets to download and explore data |1|
|--> --> Becoming one with data |1|
|--> Creating a pre-processing function for our data |1|
|--> Batching and preparing datasets for modelling |1|
|--> Setting up callbacks and mixed-precision training (faster model training) |1|
|--> Building the feature extraction model |1|
|--> loading and evaluating model using checkpoint weights |1|
|--> Creating function to get top-k accuracy using test dataset |1|
|--> Preparing model layers for fine-tuning |1|
|--> `EarlyStopping` callback and `ReduceLRonPlateau()` callback setup |1|
|--> Fine-tuning feature extraction model to beat Deep Food paper |1|
|--> Evaluating model and comparing with DeepFood |1|


| <u>**10. Natural Language Processing using Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Introduction to NLP fundementals |1|
|--> Visualizing a text dataset |1|
|--> Splitting data into training and validation set |1|
|--> Converting text to numbers using tokenization and embedding them |1|
|--> Modelling a text-dataset (running series of experiments) |1|
|--> Model 0: creation and evaluation |1|
|--> Creating an evaluation metric function |1|
|--> Model 1: [Feed forward Neural Network (dense model)]: creation and evaluation |2|
|--> Visualizing our model's learned word embeddings |2|
|--> Recurrent Neural Networks (RNNs)|3|
|--> Model 2: [LSTM (RNN)] |3|
|--> Model 3: [GRU (RNN)] |4|
|--> Model 4: [Bidirectional LSTM (RNN)] |5|
|--> Intuition behind 1D CNN |6|
|--> Model 5: [1D CNNs] |6|
|--> Model 6: Tensorflow hub pre-trained feature extractor |7|
|--> Model 7: Tensorflow hub pre-trained feature extractor (10% of data) |8|
|--> Compare all the modelling experiments |9|
|--> Finding and Visualizing best-model's most wrong predictions |10| 
|--> Making predictions on test-dataset |10| 
|--> Speed/Score Tradeoff |10|

| <u>**11. Time Series using Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Data loading and pre-processing|1|
|--> Alternative to import data using python-csv module |1|
|--> Creating train-test splits for time-series data (wrong-way)|1|
|--> Creating train-test splits for time-series data (right-way)|1|
|--> Creating plotting function to visualize time-series data |1|
|--> Modelling Experiments |2|
|--> Building Naive model (baseline) |2|
|--> Most common time series evaluation metrics |2|
|--> Implementing MASE in code |2|
|--> Creating function to evaluate model's forcast |2|
|--> Other models for time series forcasting|2|
|--> Windowing our dataset|3|
|--> Preprocessing function to window our dataset|3|
|--> Turning windows into training and test sets|3|
|--> Creating a modelling checkpoint callback to save our model|4|
|--> Building Dense model (model 1) (horizon = 1, window = 7)|4|
|--> Building Model 2 (same as model 1) (horizon = 1, window = 30)|5|
|--> Building Model 3 (same as model 1) (horizon = 7, window = 30)|6|
|--> Comparing modelling experiments (baseline,1,2,3)|7|
|--> Preparing data for input to sequence models |8|
|--> Building Model 4 (conv1D) (horizon = 1, window = 7)|8|
|--> Building Model 5 (LSTM) (horizon = 1, window = 7)|9|
|--> Multivariate time series (Model 6)|10|
|--> Making a windowed dataset using Pandas|10|
|--> Replicating N-BEATS algorithm (Model 7)|11|
|--> --> Building and testing the N-BEATS block layer|11|
|--> --> Creating a performant data pipeline using `tf.data` |11|
|--> --> Setting up hyperparameters for N-BEATS (Generic version) with Daily data |11|
|--> --> Setting up residual connections in N-BEATS |11|
|--> --> Building, Compiling and fitting the N-Beats algorithm |11|
|--> --> Saving and loading N-BEATS model |11|
|--> --> Plotting the N-BEATS architecture  |11|
|--> Creating an ensemble (Model 8) |12|
|--> Saving and loading trained ensemble model |12|
|--> Importance of prediction intervals (uncertainity estimates) in forcasting |12|
|--> Types of uncertainity in machine learning |13|
|--> Future Prediction Model (Model 9)|13|
|--> Black Swan Theory - The turkey problem (Model 10) (same as model 1) (horizon = 1, window = 7) |14|
|--> Comparing the models trained so far |15|