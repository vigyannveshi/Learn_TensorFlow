### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**4. Neural Network Classification with Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Steps in modelling classification model with tensorflow |[01](01_neural_network_classifier.ipynb)|
|02. Using history of `model.fit()` |[01](01_neural_network_classifier.ipynb)|
|03. Finding the best learning rate |[02](02_neural_network_classifier.ipynb)|
|04. Classification Evaluation Methods |[03](03_neural_network_classifier.ipynb)|
|05. Multiclass classification         |[04](04_neural_network_classifier.ipynb)|
|06. Improving Predictions on Fashion MNIST using normalized data |[05](05_neural_network_classifier.ipynb)|
|07. Finding ideal learning rate for Fashion MNIST |[06](06_neural_network_classifier.ipynb)|
|08. Evaluating trained model on Fashion MNIST |[06](06_neural_network_classifier.ipynb)|
|09. Understanding patterns learnt by our model|[07](07_neural_network_classifier.ipynb)|



### Notes (Introduction)

* Binary Classification / Multiclass Classification / Multilabel Classification
* 32 is the default batch size in tensorflow
* For Multiclass classification, shape of output --> no of classes 
* Output activation 
    --> Usually sigmoid for binary classification
    --> Usually softmax for multiclass classification
* Loss function 
    --> Usually binary cross entropy for binary classification
    --> Usually categorical cross entropy for multiclass classification
* THe combination of linear (straight lines) and non-linear (non-straight lines) functions is one of the key fundementals of neural networks