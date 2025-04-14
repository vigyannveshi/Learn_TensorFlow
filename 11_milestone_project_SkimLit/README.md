### Content

| <u>**11. Capstone - Project: SkimLit**</u>  ||
|---------|----------|
| **Concept** | **Notebook/Scripts** |
|--> Data loading and pre-processing |[01](01_skimlit.ipynb)|
|--> --> How we want our data to look |[01](01_skimlit.ipynb)|
|--> --> Getting list of sentences |[01](01_skimlit.ipynb)|
|--> --> Making numeric labels |[01](01_skimlit.ipynb)|
|--> Experiments to run |[01](01_skimlit.ipynb)|
|--> Model 0: Naive Bayes with TF-IDF (baseline)|[01](01_skimlit.ipynb)|
|--> Preparing data for deep sequence models |[02](02_skimlit.ipynb)|
|--> Model 1: Conv1D with custom token embeddings|[02](02_skimlit.ipynb)|
|--> Using pretrained embedding layer [USE (Universal Sentence Encoder)] |[03](03_skimlit.ipynb)|
|--> Model 2:  Pretrained token embedding: USE-embedding layer + Dense layer |[03](03_skimlit.ipynb)|
|--> Creating character-level tokenizer |[04](04_skimlit.ipynb)|
|--> Creating character-level embedding layer |[04](04_skimlit.ipynb)|
|--> Model 3: Conv1D with character embeddings |[04](04_skimlit.ipynb)|
|--> Model 4: Multi-modal input model with Pretrained token embeddings (same as 2) + character embedding (same as 3)|[05](05_skimlit.ipynb)|
|--> Preparing dataset for multimodal data |[05](05_skimlit.ipynb)|
|--> Encoding the line number feature to be used with Model 5 |[06](06_skimlit.ipynb)|
|--> Encoding the total lines feature to be used with Model 5 |[06](06_skimlit.ipynb)|
|--> Model 5: Multi-modal input model with Pretrained token embeddings (same as 2) + character embedding (same as 3) + positional embeddings|[06](06_skimlit.ipynb)|
|--> Compile Model 5 with label-smoothing|[06](06_skimlit.ipynb)|
|--> Saving and loading the best-performing model |[06](06_skimlit.ipynb)|
|--> Comparing the performance of all the models |[06](06_skimlit.ipynb)|
|--> Creating an end to end pipeline to input abstract and get output classified text|[07](07_skimlit.ipynb)|
|--> Python program to skim-through PUBMED-RCT abstracts|[main.py](main.py)|

### Breaking abstract into sections
* Run `python3 main.py --filename "yourfile.txt"`
* Output will be available in `output.txt`

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