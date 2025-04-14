### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**10. Natural Language Processing using Tensorflow**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Introduction to NLP fundementals |[01](01_nlp_using_tensorflow.ipynb)|
|02. Visualizing a text dataset |[01](01_nlp_using_tensorflow.ipynb)|
|03. Splitting data into training and validation set |[01](01_nlp_using_tensorflow.ipynb)|
|04. Converting text to numbers using tokenization and embedding them |[01](01_nlp_using_tensorflow.ipynb)|
|05. Modelling a text-dataset (running series of experiments) |[01](01_nlp_using_tensorflow.ipynb)|
|06. Model 0: creation and evaluation |[01](01_nlp_using_tensorflow.ipynb)|
|07. Creating an evaluation metric function |[01](01_nlp_using_tensorflow.ipynb)|
|08. Model 1: [Feed forward Neural Network (dense model)]: creation and evaluation |[02](02_nlp_using_tensorflow.ipynb)|
|09. Visualizing our model's learned word embeddings |[02](02_nlp_using_tensorflow.ipynb)|
|10. Recurrent Neural Networks (RNNs)|[03](03_nlp_using_tensorflow.ipynb)|
|11. Model 2: [LSTM (RNN)] |[03](03_nlp_using_tensorflow.ipynb)|
|12. Model 3: [GRU (RNN)] |[04](04_nlp_using_tensorflow.ipynb)|
|13. Model 4: [Bidirectional LSTM (RNN)] |[05](05_nlp_using_tensorflow.ipynb)|
|14. Intuition behind 1D CNN |[06](06_nlp_using_tensorflow.ipynb)|
|15. Model 5: [1D CNNs] |[06](06_nlp_using_tensorflow.ipynb)|
|16. Model 6: Tensorflow hub pre-trained feature extractor [USE (Universal Sentence Encoder)] |[07](07_nlp_using_tensorflow.ipynb)|
|17. Model 7: Tensorflow hub pre-trained feature extractor (10% of data) |[08](08_nlp_using_tensorflow.ipynb)|
|18. Compare all the modelling experiments |[09](09_nlp_using_tensorflow.ipynb)|
|19. Finding and Visualizing best-model's most wrong predictions |[10](10_nlp_using_tensorflow.ipynb)| 
|20. Making predictions on test-dataset |[10](10_nlp_using_tensorflow.ipynb)| 
|21. Speed/Score Tradeoff |[10](10_nlp_using_tensorflow.ipynb)|



### Notes (Introduction)
* Natural Language Processing (NLP) & Natural Language Understanding (NLU)
* In NLP words are refered to tokens
* Sequence to Sequence problems:
<img src='https://karpathy.github.io/assets/rnn/diags.jpeg'/>
  * One to one 
  * One to many [image-captioning]
  * Many to one [sentiment-analysis, time-series forcasting]
  * Many to many (unsynchronized) [machine-translation]
  * Many to many (synchronized) 

* Example inputs and outputs of NLP problem
  inputs shape: (batch_size, embedding_size)
  output shape: depends on our application  

* Applications:
  * Classification problem (eg: What tags should this article have? [multilabel options per sample])
  * Text Generation 
  * Machine Translation
  * Voice assistants

* Typical architecture of RNN
  * Input
  * Text vectorization
  * Embeddings
  * RNN cell(s)
  * Hidden activation
  * Pooling layer
  * Fully Connected layer
  * Output layer
  * Output activation
