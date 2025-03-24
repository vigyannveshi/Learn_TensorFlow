### Content

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
