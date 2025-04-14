### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**5. Convolutional Neural Network & Computer Vision with Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Building the first CNN (baseline model)|[01](01_cnn_cv_tensorflow.ipynb)|
|02. Overcoming overfitting |[02](02_cnn_cv_tensorflow.ipynb)|
|03. Data augmentation |[02](02_cnn_cv_tensorflow.ipynb)|
|04. Making a prediction with our trained model on custom data |[02](02_cnn_cv_tensorflow.ipynb)|
|05. Multiclass image classification |[03](03_cnn_cv_tensorflow.ipynb)|
|06. Visualizing each layer and activation output |[03](03_cnn_cv_tensorflow.ipynb)|
|07. Adjust model hyperparams to reduce overfitting / beat the baseline model |[04](04_cnn_cv_tensorflow.ipynb),[05](05_cnn_cv_tensorflow.ipynb),[06](06_cnn_cv_tensorflow.ipynb),[07](07_cnn_cv_tensorflow.ipynb)|
|08. Making a prediction with our trained model on custom data (multi-class) |[07](07_cnn_cv_tensorflow.ipynb)|
|09. Saving and loading our multi-class model |[07](07_cnn_cv_tensorflow.ipynb)|



### Notes:

* General steps in modelling
  * Become one with the data
    * visualize,visualize,visualize
  * Preprocess the data (get it ready for  a model)
  * Create a model (start with a baseline)
  * Evaluate the model
  * Adjust different hyperparameters and improve the model (try to beat baseline / reduce overfitting)
  * Repeat until satisfied

* General steps to reduce overfitting
  * *Get more data*: having more data gives a model more opportunity to learn diverse patterns
  * *Simplifiy the model*: if our current model is overfitting the data, it may be too complex
    * Reducing number of layers
    * Reducing number of hidden units per layer
  * *Data augmentation*: manipulates training data in such a way to add more diversity to it (without altering the original data)
  * *Transfer learning*: It leverages the patterns another model has learned on similar data as your own and allows to use those patterns on your own dataset.

* Some methods to improve model:
  * by runnning lots of experiments, namely
    * reconstructing model's architecture (increasing layers/hidden units)
    * adjust learning rate
    * try different methods of data augmentation (adjust hyperparams of ImageDataGenerator)
    * train for longer 
    * Using transfer learning

