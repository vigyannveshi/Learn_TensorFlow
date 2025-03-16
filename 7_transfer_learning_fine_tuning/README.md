### Content

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



### Notes:
* Keras.Sequential() vs Keras.Functional()
  * The Functional() API is more flexible and is able to produce more sophisticated models compared to Sequential() API