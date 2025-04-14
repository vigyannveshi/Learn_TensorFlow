# DL needs
import tensorflow as tf
import keras as kr
import tensorflow_hub as hub

# Numerical computation needs
import numpy as np

# pre-processing needs 
import re

# file handling
import argparse
import os
path = os.getcwd()

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


@kr.saving.register_keras_serializable(package="my_custom_package")
class UniversalEncodedLayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.use_layer = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2",
                                        input_shape = [],
                                        dtype=tf.string,
                                        trainable=False, # default=False,
                                        name='USE'
                                        )

    def call(self,inputs):
        return self.use_layer(inputs)

class AbstractNotFoundError(Exception):
    def __init__(self, message="Abstract not found or invalid lines provided."):
        self.message = message
        super().__init__(self.message)

class SkimLit:
    global path
    def __init__(self,model_file = f'{path}/models/skimlit.keras'):
        self.class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
        self.model = tf.keras.models.load_model(model_file,custom_objects={'USE':UniversalEncodedLayer})
        pass

    def split_chars(self,text):
        return " ".join(list(text))
    
    def replace_numbers_and_strip(self,text):
        # Replace any sequence of digits (possibly with decimal points) with @
        return re.sub(r'\d+(\.\d+)?', '@', text).strip()

    def get_abstract_lines(self,filename):
        # read the data
        with open(filename,'r') as file:
            data = file.read()

        # Use regex to split
        ### logic: New line is whenever we encounter a '.' punctuation followed by a uppercase letter
        lines = re.split(r'(?<=[.])\s+(?=[A-Z])', data)
        return lines

    def preprocess(self,lines,seq_len_line_nums = 15, seq_len_total_lines = 20):
        lines = [self.replace_numbers_and_strip(line) for line in lines]

        test_sentences = tf.constant([line for line in lines],dtype=tf.string)
        test_chars = tf.constant([self.split_chars(line) for line in lines],dtype=tf.string)
       
        # line_numbers and total_lines need one-hot-encoding
        test_line_numbers = tf.constant(np.array([tf.one_hot(line_num,depth=seq_len_line_nums) for line_num in range(len(lines)) ]),dtype=tf.float32)
        test_total_lines = tf.constant(np.array([tf.one_hot(len(lines),depth = seq_len_total_lines)]*len(lines)),dtype=tf.float32)
        return test_sentences,test_chars,test_line_numbers,test_total_lines

    def classify(self,test_input):
        predictions =  self.model.predict(test_input,verbose = 0)
        pred_labels = tf.argmax(predictions,axis=1)
        return pred_labels

    def skim_abstract(self,filename = None,lines = None):
        if filename:
            lines = self.get_abstract_lines(filename)
        if not lines:
            raise AbstractNotFoundError()

        test_sentences,test_chars,test_line_numbers,test_total_lines = self.preprocess(lines)
        labels = self.classify((test_sentences,test_chars,test_line_numbers,test_total_lines))

        output_dict = {class_name:"" for class_name in self.class_names}

        for line_num,line in enumerate(lines):
            output_dict[self.class_names[labels[line_num]]]+= line.strip() + ' '

        output = ""
        output += f"BACKGROUND: {output_dict['BACKGROUND']}\n\n" if output_dict['BACKGROUND'] else ""
        output += f"OBJECTIVE: {output_dict['OBJECTIVE']}\n\n" if output_dict['OBJECTIVE'] else ""
        output += f"METHODS: {output_dict['METHODS']}\n\n" if output_dict['METHODS'] else ""
        output += f"RESULTS: {output_dict['RESULTS']}\n\n" if output_dict['RESULTS'] else ""
        output += f"CONCLUSIONS: {output_dict['CONCLUSIONS']}\n\n" if output_dict['CONCLUSIONS'] else ""

        with open(f'{path}/output.txt','w') as file:
            file.write(output)


# GUI Application
class SkimLitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SkimLit - PUBMED-RCT Abstract Classifier")
        self.root.geometry("1000x1000")
        
        self.skimlit = SkimLit()
        self.data = None
        self.filename = None
        
        self.label = tk.Label(root, text="Select Abstract File", font=('Arial', 14))
        self.label.pack(pady=10)
        
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file, font=('Arial', 12))
        self.browse_button.pack(pady=5)
        
        ### PASTING ABSTRACT FUNCTIONALITY
        # Label for 'OR'
        self.or_label = tk.Label(root, text="OR", font=('Arial', 12))
        self.or_label.pack(pady=5)

        # Label for paste instructions
        self.paste_label = tk.Label(root, text="Paste Abstract Below:", font=('Arial', 14))
        self.paste_label.pack(pady=5)

        # Text area for pasting abstract
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=('Arial', 12))
        self.text_area.pack(pady=5)

        self.process_button = tk.Button(root, text="Skim through Abstract", command=self.process_abstract, font=('Arial', 12))
        self.process_button.pack(pady=10)

        self.clear_button = tk.Button(root, text="Clear All", command=self.clear_contents, font=('Arial', 12))
        self.clear_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Classified Abstract Output:", font=('Arial', 14))
        self.result_label.pack(pady=5)

        self.result_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=('Arial', 12))
        self.result_area.pack(pady=5)

    def clear_contents(self):
        self.filename = None
        self.data = None
        self.result_area.delete('1.0', tk.END)
        self.text_area.delete('1.0', tk.END)


    def browse_file(self):
        self.filename = filedialog.askopenfilename(title="Select Abstract File", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if self.filename:
            messagebox.showinfo("Selected File", f"File Selected: {self.filename}")
            with open(self.filename, 'r') as file:
                self.data = file.read()
                self.text_area.delete('1.0', tk.END)
                self.text_area.insert(tk.END, self.data)

    def process_abstract(self):
        # Prefer pasted data if any; else use selected file
        pasted_text = self.text_area.get('1.0', tk.END).strip()
        
        if pasted_text:
            lines = re.split(r'(?<=[.])\s+(?=[A-Z])', pasted_text)
            try:
                self.skimlit.skim_abstract(lines=lines)
                with open(f'{path}/output.txt', 'r') as file:
                    output_data = file.read()
                self.result_area.delete('1.0', tk.END)
                self.result_area.insert(tk.END, output_data)
            except AbstractNotFoundError as e:
                messagebox.showerror("Error", str(e))
        elif self.filename:
            try:
                self.skimlit.skim_abstract(filename=self.filename)
                with open(f'{path}/output.txt', 'r') as file:
                    output_data = file.read()
                self.result_area.delete('1.0', tk.END)
                self.result_area.insert(tk.END, output_data)
            except AbstractNotFoundError as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showwarning("Warning", "No abstract provided. Please upload a file or paste an abstract.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SkimLitApp(root)
    root.mainloop()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="SkimLit: Classify PUBMED-RCT abstracts into sections")
#     parser.add_argument('--filename', type=str, required=True, help='Path to the abstract file')

#     args = parser.parse_args()

#     sk = SkimLit()
#     sk.skim_abstract(os.path.join(path, args.filename))


