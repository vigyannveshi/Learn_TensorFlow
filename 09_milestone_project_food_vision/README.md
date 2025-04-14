### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**9. Capstone - Project: Food Vision**</u>  ||
|---------|----------|
| **Concept** | **Notebook/Script** |
|01. Using Tensorflow datasets to download and explore data |[01](01_food_vision_capstone_project.ipynb)|
|01.1 Becoming one with data |[01](01_food_vision_capstone_project.ipynb)|
|02. Creating a pre-processing function for our data |[01](01_food_vision_capstone_project.ipynb)|
|03. Batching and preparing datasets for modelling |[01](01_food_vision_capstone_project.ipynb)|
|04. Setting up callbacks and mixed-precision training (faster model training) |[01](01_food_vision_capstone_project.ipynb)|
|05. Building the feature extraction model |[01](01_food_vision_capstone_project.ipynb)|
|06. loading and evaluating model using checkpoint weights |[01](01_food_vision_capstone_project.ipynb)|
|07. Creating function to get top-k accuracy using test dataset |[01](01_food_vision_capstone_project.ipynb)|
|08. Preparing model layers for fine-tuning |[01](01_food_vision_capstone_project.ipynb)|
|09. `EarlyStopping` callback and `ReduceLRonPlateau()` callback setup |[01](01_food_vision_capstone_project.ipynb)|
|10. Fine-tuning feature extraction model to beat Deep Food paper |[01](01_food_vision_capstone_project.ipynb)|
|11. Evaluating model and comparing with DeepFood |[01](01_food_vision_capstone_project.ipynb)|
|12. Creating an end to end pipeline to input a food image and get its name |[02](02_food_vision_capstone_project.ipynb)|
|##. Creating an GUI interface of Food-Vision to upload a food picture and get its name |[main.py](main.py)|

### Running Food Vision
* Unzip [`models.zip`](09_milestone_project_food_vision/models.zip)
* Open a terminal in `09_milestone_project_food_vision` folder and run: `python3 main.py`
* A GUI will open up where you can add images from your local system or paste a image-address from the web.
* You will be notified with the food name.
