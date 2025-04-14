### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**6. Transfer learning - feature extraction**</u>  ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Setting up callbacks |[01](01_transfer_learning_feature_extraction.ipynb)|
|02. Creating models using Tensorflow Hub |[01](01_transfer_learning_feature_extraction.ipynb)|
|03. Building and Compiling Tensorflow feature extraction model |[01](01_transfer_learning_feature_extraction.ipynb)|
|04. Resnet-50 v2 vs Efficient net b0 |[01](01_transfer_learning_feature_extraction.ipynb)|
|05. Visualization using TensorBoard |[01](01_transfer_learning_feature_extraction.ipynb)|


### Notes:
* Transfer Learning: Model learns patterns/insights from similar problem space, and patterns get used/tuned to specific problem
* Can leverage an existing NN architecture proven to work on problems similar to our own
* Can leverage a working neural network architecture which has already learned patterns on similar data to our own (often results in great results in less data)
* **Applications:**
  * Computer Vision
    * ImageNet dataset SOTA models (EfficientNet Architecture [already works really well on CV tasks])
  * Natural Language Processing
  * ...etc

* **Types of Transfer learning:**
  * `As is` transfer learning - using existing transfer learning - with no changes whatsoever (Using ImageNet model on 1000 ImageNet classes, none of your own) [even the output layer is not changed]
  * `Feature Extraction` transfer learning - use the pre-learned patterns of an existing model (eg: efficientnet b0 trained on ImageNet) and adjust the output layer for your own problem (eg. 1000 classes of ImageNet --> 10 classes of food)
  * `Fine Tuning` transfer learning - use the pre-learned patterns of an existing model and "fine-tune" many or all of the underlying layers (including new output layers) [typically requires more training data]