### Content

| <u>**4. Neural Network Classification with Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|--> Steps in modelling classification model with tensorflow |1|
|--> Using history of `model.fit()` |1|
|--> Finding the best learning rate |2|
|--> Classification Evaluation Methods |3|





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