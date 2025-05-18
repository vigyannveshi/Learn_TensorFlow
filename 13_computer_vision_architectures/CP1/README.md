### ↩️ [**Computer Vision Architectures using Tensorflow**](../README.md)

### **Replicating QUICKSAL**
  <!-- |BottleNeck Inverted Residual (BIR) Block|Implementated (BIR) Block|
  |---|---|
  |<img src = "BIR.png"/>|<img src = "bir_block.png" height = 500/>|

  <br>
  <br>

  |Inception Block|Implementated Inception Block|
  |---|---|
  |<img src = "INCEPTION_BLOCK.png"/>|<img src = "inception_block.png" width = 800/>|

  <br>
  <br>

  |QUICKSAL|Implementated QUICKSAL|
  |---|---|
  |<img src = "QUICKSAL.png"/>|<img src = "quicksal_vis.png" width = 900/>| -->

  <!-- Bottleneck Inverted Residual (BIR) Block -->
<b>➤ Bottleneck Inverted Residual (BIR) Block</b>
<table border="1" cellspacing="0" cellpadding="10">
  <tr>
    <th>BottleNeck Inverted Residual (BIR) Block</th>
    <th>Implemented (BIR) Block</th>
  </tr>
  <tr>
    <td><img src="BIR.png"/></td>
    <td><img src="bir_block.png" height="500"/></td>
  </tr>
</table>

<br><br>

<!-- Inception Block -->
<b>➤ Inception Block</b>
<table border="1" cellspacing="0" cellpadding="10">
  <tr>
    <th>Inception Block</th>
    <th>Implemented Inception Block</th>
  </tr>
  <tr>
    <td><img src="INCEPTION_BLOCK.png"/></td>
    <td><img src="inception_block.png" width="800"/></td>
  </tr>
</table>

<br><br>

<!-- QUICKSAL -->
<b>➤ QUICKSAL</b>
<table border="1" cellspacing="0" cellpadding="10">
  <tr>
    <th>QUICKSAL</th>
    <th>Implemented QUICKSAL</th>
  </tr>
  <tr>
    <td><img src="QUICKSAL.png"/></td>
    <td><img src="quicksal_vis.png" width="900"/></td>
  </tr>
</table>


### Content

| <u>**CP1: Replicating QUICKSAL**</u> ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Building the Bottleneck inverted residual block (BIR)  |[01](01_cp1.ipynb)|
|02. Creating the encoder section  |[01](01_cp1.ipynb)|
|03. Adding pre-trained weights of MobileNet-v2 (imagenet) to encoder layer, and setting `trainable=False` except the expand layer in the first block |[01](01_cp1.ipynb)|
|04. Building the inception block |[02](02_cp1.ipynb)|
|05. Decoder creation |[02](02_cp1.ipynb)|
|06. Putting all blocks together to build QUICKSAL|[03](03_cp1.ipynb)|
|07. Testing model with an image from MSRA dataset|[03](03_cp1.ipynb)|
|08. Preparing the data to be loaded into model|[04](04_cp1.ipynb)|
|09. Importance of `tf.py_function` to run custom python function in middle of tensorflow pipeline|[04](04_cp1.ipynb)|
|10. Creating efficient data loader using `tf.data`|[04](04_cp1.ipynb)|
|11. Creating a python script with the QUICKSAL model implementations |[models/model.py](models/model.py)|
|12. Running a trial fit() for 1 epoch|[04](04_cp1.ipynb)|
|13. Creating callbacks, allow mixed precision training and compile the model |[05](05_cp1.ipynb)|
|14. Fitting the model|[05](05_cp1.ipynb)|
|15. Improving the encoder-section to incorporate MobileNet-V2 initial layers|[06](06_cp1.ipynb)|
|16. Encorporating encoder into QUICKSAL|[06](06_cp1.ipynb)|
|17. Visualizing the model |[07](07_cp1.ipynb)|




### Notes (Introduction)
* We will attempt to replicate the works from the paper -  [QUICKSAL: A small and sparse visual saliency model for efficient inference in resource constrained hardware](https://labs.dese.iisc.ac.in/neuronics/wp-content/uploads/sites/16/2020/02/0847_Final.pdf)
* Approach to replicate the paper:
  * The paper uses a network architecture which comprises fundementally of two major blocks:
    1. Bottle Neck inverted residual block (taken from the paper: [MobileNet V2](https://arxiv.org/pdf/1801.04381)) {some kind of feature extractor / encoder}
    2. Inception Block (inspired from [SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS](https://arxiv.org/pdf/1412.7062), [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842))
  * We will create a custom layer/ block for each of this blocks
  * The paper also gives the complete architecture of QUICKSAL, which we will be creating combining this created custom blocks.
  * The paper uses weights from MobileNet V2 for BottleNeck layers {encoder}, which they order in a particular architecture, we shall stick to it.
  * Evaluation metrics: (Mean Absolute Error (MAE), Weighted $F_{\beta}$ measure, with $\beta^2 = 0.3$)<br>
  <br>
  $MAE = \frac{1}{H \cdot W} \cdot \sum_{i,j}{|G(x_{i,j}) - P(x_{i,j})|}$
  <br>
  <br>
  $F_{\beta} = \frac{(1+\beta^2) \cdot Precision \cdot Recall}{(\beta^2) \cdot Precision + Recall}$
  <br>
  * The loss function is not explicitly mentioned in the paper. We will use Mean Absolute Error (MAE) as a loss function, considering it is a common metric used also to compare the model with the other SOTA available.
  * They have used Adam optimizer with CyclicLR schedular {base LR = 1e-4, max LR = 1e-3}
  * They have trained the model on MSRA10K dataset with train/val/test split of 0.8/0.1/0.1 respectively. This can be downloaded from: http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip
  * We allow mixed-precision training to speed up training.
  * We use [`ModelCheckpoint(save_best_only = True, save_weights_only = False)`,<br> `EarlyStopping(patience = 10, monitor = 'val_loss')`,<br> `CyclicLR(base_lr=1e-4, max_lr=1e-3, step_size=2000)`] callbacks.
  * Leslie Smith (who invented CyclicLR) recommends to set step_size between 2× and 10× the number of batches in one epoch. Reference: [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186)
  * In our case batches per epoch = 8000/32 = 250,
   
      |Option|	Step Size|
      |---|---|
      |Small cycle	2 × 250 |500 batches|
      |Medium cycle	4 × 250 | 1000 batches|
      |Big cycle	8 × 250   | 2000 batches|
  
  * Choosing step_size= 2000 
    * 2000 is ~8 epochs.
    * So LR slowly rises over 8 epochs, then slowly falls over 8 epochs.
    * This is very stable and gentle → perfect for small datasets like MSRA with only 8k images.
    * Also stops overfitting — doesn't jerk the LR up/down too fast.
  
  * We will fit the model for 50 epochs
  * **Important Note: Changes implemented from notebook 3 onwards (including the models/model.py)**
    * We have replaced `tf.keras.layers.ReLU` with `tf.keras.layers.Activation('relu')` to ensure that its dtype is set `tf.float32`.
    * We have set the dtype of `tf.keras.layers.Activation('softmax')` to `tf.float32`.
    * This is done so that during mixed precision training, the precision of activation function output remains float32.
    * The encoder-section of model is not trainable except for block-1 expand layer in block 1 of encoder. This has to be trained because, we couldn't find pre-trained weights corresponding to this layer in the MobileNet-V2 pretrained model.
  * After trying to fit the model with the setup done upto (14.) it was observed that the model is not learning at all (tried for 12 epochs) the val_loss didn't change. 
    * Possible Reasons:
      1. We have used custom expansion conv layer in first block, and since it is not trained overall prediction and learning is hampered?
      * Soln: We will use the mobilenet-v2 model as it is (including the first layer) and improve the encoder, here the encoder is some-kind of feature extractor model.
      * Attempted the above solution and tried fitting, no change at all, still model not learning.
       
      1. May be too much data to train?
      * Soln: Use only 10% train and 10% val data, and check if we can overfit the model
      * Attempted the above solution and tried fitting, model still not learning.

<<<<<<< HEAD
      1. Vanishing gradient problem?
      * Soln: Use a scaled MAE loss. Scale MAE but 1e3
=======
      3. Vanishing gradient problem?
      * Soln: Use a scaled MAE loss. Scale MAE by 1e3
>>>>>>> 40851d3a874bbc9a4fcac8b1beed4b5bd8e68586
      * Attempted this, but no-GPU time remaining on colab, GPU running extremely slow hence can't verify.
      
      1. May be connections in model are not correct?
      * Solution: Visualise the entire model - every block 
      * Visualized model, model created as expected.
    * Most probable reason --> some kind of vanishing gradients or  needs a better loss function
