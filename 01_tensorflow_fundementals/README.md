### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**1. Tensorflow Fundamentals**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
| 01. Creating tensors using `tf.constant()` |[01](01_tensorflow_fundementals.ipynb)|
| 02. Creating tensors using `tf.Variable()` |[01](01_tensorflow_fundementals.ipynb)|
| 03. Creating random tensors using `tf.random.distribution()` |[01](01_tensorflow_fundementals.ipynb)|
| 04. Shuffle the order of elements of tensor using `tf.random.shuffle()` |[02](02_tensorflow_fundementals.ipynb)|
| 05. Other ways to create tensors |[03](03_tensorflow_fundementals.ipynb)|
| 06. Getting information from tensors |[04](04_tensorflow_fundementals.ipynb)|
| 07. Indexing and expanding tensors |[05](05_tensorflow_fundementals.ipynb)|
| 08. Reshaping the tensor using `tf.reshape()` |[05](05_tensorflow_fundementals.ipynb)|
| 09. Adding an extra dimension to tensor using `tf.newaxis` or `tf.expand_dims()` |[05](05_tensorflow_fundementals.ipynb)|
| 10. Manipulating tensors with basic operations (+,-,*,/) |[06](06_tensorflow_fundementals.ipynb)|
| 11. Matrix Multiplication |[07](07_tensorflow_fundementals.ipynb)|
| 12. Dot Product |[07](07_tensorflow_fundementals.ipynb)|
| 13. Changing datatype of tensors using `tf.cast()` |[08](08_tensorflow_fundementals.ipynb)|
| 14. Aggregating tensors |[09](09_tensorflow_fundementals.ipynb)|
| 15. Finding positional maximum and minimum |[10](10_tensorflow_fundementals.ipynb)|
| 16. Squeezing a tensor `tf.squeeze()` |[11](11_tensorflow_fundementals.ipynb)|
| 17. One Hot Encoding `tf.one_hot(indices:tensor,depth:"number of classes")` |[12](12_tensorflow_fundementals.ipynb)|
| 18. Some more tensor functions and operations |[13](13_tensorflow_fundementals.ipynb)|
| 19. Tensors and Numpy |[14](14_tensorflow_fundementals.ipynb)|
| 20. Understanding Devices to speed up TensorFlow |[15](15_tensorflow_fundementals.ipynb)|
| ##. Exercise |[16](16_tensorflow_fundementals.ipynb)|



### Notes (Introduction):
* ML can be used for literally anything as long as you can convert it into numbers and program it to find patterns. Literally it could be any input or output from the universe. 

* ML algorithms like Random Forest, Naive Bayes, Nearest Neighbour, Support Vector machine.... many more are refered to as "shallow algorithms" since the advent of Deep Learning.

* Terms used for patterns:<br>
    1. embeddings
    2. weights 
    3. feature representation
    4. feature vectors

* Sound to text: Sequence to Sequence (Seq to Seq) deep learning problems.

* deep learning architectures <--> deep learning models.

### Notes (Tensorflow):
* Tensor: Numeric way to represent information.

* One of the best explainations to what a tensor is:
https://www.youtube.com/watch?v=f5liqUk0ZTw

* A vector can represent an area, make the length of the vector proportional to amount of the area (if v1 and v2 are vectors in the areal surface, eg: vectors along adjacent sides of a parallelogram their cross product is a vector perpendicular to the area, and length of vector is proportional to area)


* Tensorflow Workflow:
    1. Get data ready (turn into tensors)
    2. Build or pick a restrained model
    3. Fit the model to the data and make a prediction
    4. Evaluate the model 
    5. Improve through experimentation
    6. Save and reload your trained model