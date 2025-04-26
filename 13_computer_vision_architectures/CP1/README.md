### ↩️ [**Computer Vision Architectures using Tensorflow**](../README.md)

### Content

| <u>**CP1: Replicating QUICKSAL**</u> ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Building the Bottleneck inverted residual block (BIR)  |[01](01_cp1.ipynb)|
|02. Creating the encoder section  |[01](01_cp1.ipynb)|
|03. Adding pre-trained weights of MobileNet-v2 (imagenet) to encoder layer, and setting `trainable=False` except the expand layer in the first block |[01](01_cp1.ipynb)|
|04. Building the inception block |[02](02_cp1.ipynb)|


### Notes (Introduction)
* We will attempt to replicate the works from the paper -  [QUICKSAL: A small and sparse visual saliency model for efficient inference in resource constrained hardware](https://labs.dese.iisc.ac.in/neuronics/wp-content/uploads/sites/16/2020/02/0847_Final.pdf)
* Approach to replicate the paper:
  * The paper uses a network architecture which comprises fundementally of two major blocks:
    1. Bottle Neck inverted residual block (taken from the paper: [MobileNet V2](https://arxiv.org/pdf/1801.04381)) {some kind of feature extractor / encoder}
    2. Inception Block (inspired from [SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS](https://arxiv.org/pdf/1412.7062), [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842))
  * We will create a custom layer/ block for each of this blocks
  * The paper also gives the complete architecture of QUICKSAL, which we will be creating combining this created custom blocks.
  * The paper uses weights from MobileNet V2 for BottleNeck layers {encoder}, which they order in a particular architecture, we shall stick to it.
  * They have used Adam optimizer with CyclicLR schedular {base LR = 1e-4, max LR = 1e-3}
  * The loss function is not explicitly mentioned in the paper, we will figure that out down the line.
  * Evaluation metrics: (Mean Absolute Error (MAE), Weighted $F_{\beta}$ measure, with $\beta^2 = 0.3$)<br>
  <br>
  $MAE = \frac{1}{H \cdot W} \cdot \sum_{i,j}{|G(x_{i,j}) - P(x_{i,j})|}$
  <br>
  <br>
  $F_{\beta} = \frac{(1+\beta^2) \cdot Precision \cdot Recall}{(\beta^2) \cdot Precision + Recall}$
  <br>

  * They have trained the model on MSRA10K dataset with train/val/test split of 0.8/0.1/0.1 respectively. This can be downloaded from: http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip
