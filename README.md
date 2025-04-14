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

| <u>[**01. Tensorflow Fundamentals**](01_tensorflow_fundementals)</u>  ||
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


| <u>[**02. Neural Network Regression with Tensorflow**](02_neural_network_regression/)</u>  ||
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

| <u>[**03. Customizing Models**](03_customizing_models/)</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Custom training loop |1|
|--> Custom loss function |2|
|--> Custom model         |3|
|--> Improving Custom model |4|


| <u>[**04. Neural Network Classification with Tensorflow**](04_neural_network_classification/)</u>  ||
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


| <u>[**05. Convolutional Neural Network & Computer Vision with Tensorflow**](05_convolutional_neural_network/)</u>  ||
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

| <u>[**06. Transfer learning - feature extraction**](06_transfer_learning_feature_extraction/)</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Setting up callbacks |1|
|--> Creating models using Tensorflow Hub |1|
|--> Building and Compiling Tensorflow feature extraction model |1|
|--> Resnet-50 v2 vs Efficient net b0 |1|
|--> Visualization using TensorBoard |1|

| <u>[**07. Transfer learning - fine tuning**](07_transfer_learning_fine_tuning/)</u>  ||
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

| <u>[**08. Transfer learning - scaling up**](08_transfer_learning_scaling_up/)</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|-->Training a model to fit data of 101 classes of Food101 dataset|1|
|-->Evaluating the fitted model|1|
|-->Visualizing f1-scores for different classes|1|
|-->Checking out where our model is most wrong|1|
|-->Prediction on custom images|1|


| <u>[**09. Capstone - Project: Food Vision**](09_milestone_project_food_vision/)</u>  ||
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


| <u>[**10. Natural Language Processing using Tensorflow**](10_NLP_using_tensorflow/)</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Introduction to NLP fundementals |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|02. Visualizing a text dataset |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|03. Splitting data into training and validation set |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|04. Converting text to numbers using tokenization and embedding them |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|05. Modelling a text-dataset (running series of experiments) |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|06. Model 0: creation and evaluation |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|07. Creating an evaluation metric function |[01](10_NLP_using_tensorflow/01_nlp_using_tensorflow.ipynb)|
|08. Model 1: [Feed forward Neural Network (dense model)]: creation and evaluation |[02](10_NLP_using_tensorflow/02_nlp_using_tensorflow.ipynb)|
|09. Visualizing our model's learned word embeddings |[02](10_NLP_using_tensorflow/02_nlp_using_tensorflow.ipynb)|
|10. Recurrent Neural Networks (RNNs)|[03](10_NLP_using_tensorflow/03_nlp_using_tensorflow.ipynb)|
|11. Model 2: [LSTM (RNN)] |[03](10_NLP_using_tensorflow/03_nlp_using_tensorflow.ipynb)|
|12. Model 3: [GRU (RNN)] |[04](10_NLP_using_tensorflow/04_nlp_using_tensorflow.ipynb)|
|13. Model 4: [Bidirectional LSTM (RNN)] |[05](10_NLP_using_tensorflow/05_nlp_using_tensorflow.ipynb)|
|14. Intuition behind 1D CNN |[06](10_NLP_using_tensorflow/06_nlp_using_tensorflow.ipynb)|
|15. Model 5: [1D CNNs] |[06](10_NLP_using_tensorflow/06_nlp_using_tensorflow.ipynb)|
|16. Model 6: Tensorflow hub pre-trained feature extractor [USE (Universal Sentence Encoder)] |[07](10_NLP_using_tensorflow/07_nlp_using_tensorflow.ipynb)|
|17. Model 7: Tensorflow hub pre-trained feature extractor (10% of data) |[08](10_NLP_using_tensorflow/08_nlp_using_tensorflow.ipynb)|
|18. Compare all the modelling experiments |[09](10_NLP_using_tensorflow/09_nlp_using_tensorflow.ipynb)|
|19. Finding and Visualizing best-model's most wrong predictions |[10](10_NLP_using_tensorflow/10_nlp_using_tensorflow.ipynb)| 
|20. Making predictions on test-dataset |[10](10_NLP_using_tensorflow/10_nlp_using_tensorflow.ipynb)| 
|21. Speed/Score Tradeoff |[10](10_NLP_using_tensorflow/10_nlp_using_tensorflow.ipynb)|

| <u>[**11. Capstone - Project: SkimLit**](11_milestone_project_SkimLit/)</u>  ||
|---------|----------|
| **Concept** | **Notebook/Scripts** |
|01. Data loading and pre-processing |[01](11_milestone_project_SkimLit/01_skimlit.ipynb)|
|01.1. How we want our data to look |[01](11_milestone_project_SkimLit/01_skimlit.ipynb)|
|01.2. Getting list of sentences |[01](11_milestone_project_SkimLit/01_skimlit.ipynb)|
|01.3. Making numeric labels |[01](11_milestone_project_SkimLit/01_skimlit.ipynb)|
|02. Experiments to run |[01](11_milestone_project_SkimLit/01_skimlit.ipynb)|
|03. Model 0: Naive Bayes with TF-IDF (baseline)|[01](11_milestone_project_SkimLit/01_skimlit.ipynb)|
|04. Preparing data for deep sequence models |[02](11_milestone_project_SkimLit/02_skimlit.ipynb)|
|05. Model 1: Conv1D with custom token embeddings|[02](11_milestone_project_SkimLit/02_skimlit.ipynb)|
|06. Using pretrained embedding layer [USE (Universal Sentence Encoder)] |[03](11_milestone_project_SkimLit/03_skimlit.ipynb)|
|07. Model 2:  Pretrained token embedding: USE-embedding layer + Dense layer |[03](11_milestone_project_SkimLit/03_skimlit.ipynb)|
|08. Creating character-level tokenizer |[04](11_milestone_project_SkimLit/04_skimlit.ipynb)|
|09. Creating character-level embedding layer |[04](11_milestone_project_SkimLit/04_skimlit.ipynb)|
|10. Model 3: Conv1D with character embeddings |[04](11_milestone_project_SkimLit/04_skimlit.ipynb)|
|11. Model 4: Multi-modal input model with Pretrained token embeddings (same as 2) + character embedding (same as 3)|[05](11_milestone_project_SkimLit/05_skimlit.ipynb)|
|12. Preparing dataset for multimodal data |[05](11_milestone_project_SkimLit/05_skimlit.ipynb)|
|13. Encoding the line number feature to be used with Model 5 |[06](11_milestone_project_SkimLit/06_skimlit.ipynb)|
|14. Encoding the total lines feature to be used with Model 5 |[06](11_milestone_project_SkimLit/06_skimlit.ipynb)|
|15. Model 5: Multi-modal input model with Pretrained token embeddings (same as 2) + character embedding (same as 3) + positional embeddings|[06](11_milestone_project_SkimLit/06_skimlit.ipynb)|
|16. Compile Model 5 with label-smoothing|[06](11_milestone_project_SkimLit/06_skimlit.ipynb)|
|17. Saving and loading the best-performing model |[06](11_milestone_project_SkimLit/06_skimlit.ipynb)|
|18. Comparing the performance of all the models |[06](11_milestone_project_SkimLit/06_skimlit.ipynb)|
|19. Creating an end to end pipeline to input abstract and get output classified text|[07](11_milestone_project_SkimLit/07_skimlit.ipynb)|
|##. Python program to skim-through PUBMED-RCT abstracts|[main.py](11_milestone_project_SkimLit/main.py)|


| <u>[**12. Time Series forecasting using Tensorflow**](12_time_series_using_tensorflow/README.md)</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Data loading and pre-processing|[01](12_time_series_using_tensorflow/01_time_series_using_tensorflow.ipynb)|
|02. Alternative to import data using python-csv module |[01](12_time_series_using_tensorflow/01_time_series_using_tensorflow.ipynb)|
|03. Creating train-test splits for time-series data (wrong-way)|[01](12_time_series_using_tensorflow/01_time_series_using_tensorflow.ipynb)|
|04. Creating train-test splits for time-series data (right-way)|[01](12_time_series_using_tensorflow/01_time_series_using_tensorflow.ipynb)|
|05. Creating plotting function to visualize time-series data |[01](12_time_series_using_tensorflow/01_time_series_using_tensorflow.ipynb)|
|06. Modelling Experiments |[02](12_time_series_using_tensorflow/02_time_series_using_tensorflow.ipynb)|
|07. Building Naive model (baseline) |[02](12_time_series_using_tensorflow/02_time_series_using_tensorflow.ipynb)|
|08. Most common time series evaluation metrics |[02](12_time_series_using_tensorflow/02_time_series_using_tensorflow.ipynb)|
|09. Implementing MASE in code |[02](12_time_series_using_tensorflow/02_time_series_using_tensorflow.ipynb)|
|10. Creating function to evaluate model's forcast |[02](12_time_series_using_tensorflow/02_time_series_using_tensorflow.ipynb)|
|11. Other models for time series forcasting|[02](12_time_series_using_tensorflow/02_time_series_using_tensorflow.ipynb)|
|12. Windowing our dataset|[03](12_time_series_using_tensorflow/03_time_series_using_tensorflow.ipynb)|
|13. Preprocessing function to window our dataset|[03](12_time_series_using_tensorflow/03_time_series_using_tensorflow.ipynb)|
|14. Turning windows into training and test sets|[03](12_time_series_using_tensorflow/03_time_series_using_tensorflow.ipynb)|
|15. Creating a modelling checkpoint callback to save our model|[04](12_time_series_using_tensorflow/04_time_series_using_tensorflow.ipynb)|
|16. Building Dense model (model 1) (horizon = 1, window = 7)|[04](12_time_series_using_tensorflow/04_time_series_using_tensorflow.ipynb)|
|17. Building Model 2 (same as model 1) (horizon = 1, window = 30)|[05](12_time_series_using_tensorflow/05_time_series_using_tensorflow.ipynb)|
|18. Building Model 3 (same as model 1) (horizon = 7, window = 30)|[06](12_time_series_using_tensorflow/06_time_series_using_tensorflow.ipynb)|
|19. Comparing modelling experiments (baseline,1,2,3)|[07](12_time_series_using_tensorflow/07_time_series_using_tensorflow.ipynb)|
|20. Preparing data for input to sequence models |[08](12_time_series_using_tensorflow/08_time_series_using_tensorflow.ipynb)|
|21. Building Model 4 (conv1D) (horizon = 1, window = 7)|[08](12_time_series_using_tensorflow/08_time_series_using_tensorflow.ipynb)|
|22. Building Model 5 (LSTM) (horizon = 1, window = 7)|[09](12_time_series_using_tensorflow/09_time_series_using_tensorflow.ipynb)|
|23. Multivariate time series (Model 6)|[10](12_time_series_using_tensorflow/10_time_series_using_tensorflow.ipynb)|
|24. Making a windowed dataset using Pandas|[10](12_time_series_using_tensorflow/10_time_series_using_tensorflow.ipynb)|
|25. Replicating N-BEATS algorithm (Model 7)|[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.1. Building and testing the N-BEATS block layer|[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.2. Creating a performant data pipeline using `tf.data` |[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.3. Setting up hyperparameters for N-BEATS (Generic version) with Daily data |[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.4. Setting up residual connections in N-BEATS |[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.5. Building, Compiling and fitting the N-Beats algorithm |[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.6. Saving and loading N-BEATS model |[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|25.7. Plotting the N-BEATS architecture  |[11](12_time_series_using_tensorflow/11_time_series_using_tensorflow.ipynb)|
|26. Creating an ensemble (Model 8) |[12](12_time_series_using_tensorflow/12_time_series_using_tensorflow.ipynb)|
|27. Saving and loading trained ensemble model |[12](12_time_series_using_tensorflow/12_time_series_using_tensorflow.ipynb)|
|28. Importance of prediction intervals (uncertainity estimates) in forcasting |[12](12_time_series_using_tensorflow/12_time_series_using_tensorflow.ipynb)|
|29. Types of uncertainity in machine learning |[13](12_time_series_using_tensorflow/13_time_series_using_tensorflow.ipynb)|
|30. Future Prediction Model (Model 9)|[13](12_time_series_using_tensorflow/13_time_series_using_tensorflow.ipynb)|
|31. Black Swan Theory - The turkey problem (Model 10) (same as model 1) (horizon = 1, window = 7) |[14](12_time_series_using_tensorflow/14_time_series_using_tensorflow.ipynb)|
|32. Comparing the models trained so far |[15](12_time_series_using_tensorflow/15_time_series_using_tensorflow.ipynb)|