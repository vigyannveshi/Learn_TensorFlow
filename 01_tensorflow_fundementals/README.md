### Content

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