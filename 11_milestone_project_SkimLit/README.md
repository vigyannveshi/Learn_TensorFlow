### ↩️ [**Learn Tensorflow**](../README.md)

### Content

| <u>**11. Capstone - Project: SkimLit**</u>  ||
|---------|----------|
| **Concept** | **Notebook/Script** |
|01. Data loading and pre-processing |[01](01_skimlit.ipynb)|
|01.1. How we want our data to look |[01](01_skimlit.ipynb)|
|01.2. Getting list of sentences |[01](01_skimlit.ipynb)|
|01.3. Making numeric labels |[01](01_skimlit.ipynb)|
|02. Experiments to run |[01](01_skimlit.ipynb)|
|03. Model 0: Naive Bayes with TF-IDF (baseline)|[01](01_skimlit.ipynb)|
|04. Preparing data for deep sequence models |[02](02_skimlit.ipynb)|
|05. Model 1: Conv1D with custom token embeddings|[02](02_skimlit.ipynb)|
|06. Using pretrained embedding layer [USE (Universal Sentence Encoder)] |[03](03_skimlit.ipynb)|
|07. Model 2:  Pretrained token embedding: USE-embedding layer + Dense layer |[03](03_skimlit.ipynb)|
|08. Creating character-level tokenizer |[04](04_skimlit.ipynb)|
|09. Creating character-level embedding layer |[04](04_skimlit.ipynb)|
|10. Model 3: Conv1D with character embeddings |[04](04_skimlit.ipynb)|
|11. Model 4: Multi-modal input model with Pretrained token embeddings (same as 2) + character embedding (same as 3)|[05](05_skimlit.ipynb)|
|12. Preparing dataset for multimodal data |[05](05_skimlit.ipynb)|
|13. Encoding the line number feature to be used with Model 5 |[06](06_skimlit.ipynb)|
|14. Encoding the total lines feature to be used with Model 5 |[06](06_skimlit.ipynb)|
|15. Model 5: Multi-modal input model with Pretrained token embeddings (same as 2) + character embedding (same as 3) + positional embeddings|[06](06_skimlit.ipynb)|
|16. Compile Model 5 with label-smoothing|[06](06_skimlit.ipynb)|
|17. Saving and loading the best-performing model |[06](06_skimlit.ipynb)|
|18. Comparing the performance of all the models |[06](06_skimlit.ipynb)|
|19. Creating an end to end pipeline to input abstract and get output classified text|[07](07_skimlit.ipynb)|
|##. Python program to skim-through PUBMED-RCT abstracts|[main.py](main.py)|

### Running SkimLit
* open a terminal in [11_milestone_project_SkimLit](11_milestone_project_SkimLit) folder and run `python3 main.py` 
* A GUI will start up where in you can select a text file or paste the abstract.
* Once you add in the data, click on `Skim through Abstract` button to get the skimmed output.
* You can clear the paste area and result area by pressing the `Clear All` button.


### Notes (Introduction)
* We are trying to make medical abstracts 📄 easier to read by breaking them down into subsections: [Background 🔙, Objective 🕵️, Methods 🪜, Results 📊, Conclusion 🏁]

* It is a multi-class classification problem
  
* We will try to replicate `PubMed 200k RCT(Randomized Controlled Trials): a Dataset for Sequential Sentence Classification in Medical Abstracts`, REFERENCE: https://arxiv.org/pdf/1710.06071
  
* The model is taken from the paper: `Neural Networks for Joint Sentence Classification
in Medical Paper Abstracts`, REFERENCE: https://arxiv.org/pdf/1612.05251

* Feature Engineering: 
  * Taking non-obvious features from the data and encoding them numerically to help our model learn
  * Features are not always quite obvious.  
  * Data augmentation is a form of feature engineering, as we know the same image rotated, shifted, zoomed,shifted is the same image.
  * Engineered features need to be available at test time.