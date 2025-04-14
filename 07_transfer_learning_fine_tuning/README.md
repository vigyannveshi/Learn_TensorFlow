### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**7. Transfer learning - fine tuning**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Using `tf.keras.preprocessing.image_dataset_from_directory()` |[01](01_transfer_learning_fine_tuning.ipynb)|
|02. Performing Various Experimentations |[01](01_transfer_learning_fine_tuning.ipynb)|
|02.1. Building a baseline model for running experiments |[01](01_transfer_learning_fine_tuning.ipynb)|
|02.1.1 Building model using `Functional()` API |[01](01_transfer_learning_fine_tuning.ipynb)|
|02.2. Getting feature vector from trained model |[02](02_transfer_learning_fine_tuning.ipynb)|
|02.3. Running a model experiment using 1% of training data|[03](03_transfer_learning_fine_tuning.ipynb)|
|02.4. Running a model experiment using 10% of augmented training data|[04](04_transfer_learning_fine_tuning.ipynb)|
|02.4.1. Creating a model checkpoint callback|[04](04_transfer_learning_fine_tuning.ipynb)|
|02.4.2. Loading in model/model weights using saved checkpoints|[04](04_transfer_learning_fine_tuning.ipynb)|
|02.5. Improving the baseline model, by fine tuning weights of base model |[05](05_transfer_learning_fine_tuning.ipynb)|
|02.5.1. Loading the checkpoints of model 2 |[05](05_transfer_learning_fine_tuning.ipynb)|
|02.5.2. Fine tuning for another 5 epochs |[05](05_transfer_learning_fine_tuning.ipynb)|
|02.5.3. Comparing before and after fine-tuning |[05](05_transfer_learning_fine_tuning.ipynb)|
|02.6. Running a model experiment using 100% of augmented training data with fine tuning|[06](06_transfer_learning_fine_tuning.ipynb)|



### Notes:
* Keras.Sequential() vs Keras.Functional()
  * The Functional() API is more flexible and is able to produce more sophisticated models compared to Sequential() API