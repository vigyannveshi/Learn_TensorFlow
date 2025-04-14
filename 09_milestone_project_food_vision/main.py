# DL needs
import tensorflow as tf

# GUI
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button,Entry
from PIL import Image, ImageTk

# plotting needs
import matplotlib.pyplot as plt

# ensuring reproducibility
random_seed=42
tf.random.set_seed(random_seed)

# file handling
import argparse
import os
import requests
from io import BytesIO

path = os.getcwd()

class FoodVision:
    def __init__(self,model_filepath = f'{path}/models/food_vision.keras',labels_filepath = 'labels.txt' ):
        self.model = tf.keras.models.load_model(model_filepath)
        
        # getting labels
        with open(labels_filepath) as file:
            self.food_names = file.read().split('\n')
        pass


    def pre_process(self,img_path_or_bytes,from_local = True):
        # read image file
        if from_local:
            img_path_or_bytes = tf.io.read_file(img_path_or_bytes)
        img = tf.io.decode_image(img_path_or_bytes)

        # pre-process
        img = tf.image.resize(img,(224,224))
        img = tf.expand_dims(img,axis=0)
        return img
        
    def get_food_name(self,img_filepath,from_local = True):
        img = self.pre_process(img_filepath,from_local=from_local)
        pred = self.model.predict(img,verbose = 0)
        pred_label = tf.squeeze(tf.argmax(pred,axis = 1))
        pred_class = self.food_names[pred_label]
        return pred_class


# GUI App
class FoodVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Vision 🍕🍔🥗")
        self.root.geometry("800x700")
        
        self.fv = FoodVision()

        self.label = Label(root, text="Upload an image or paste a URL to predict", font=("Arial", 16))
        self.label.pack(pady=20)

        self.image_label = Label(root)
        self.image_label.pack()

        self.browse_button = Button(root, text="Browse Image", command=self.browse_image, font=("Arial", 12))
        self.browse_button.pack(pady=10)

        # Entry widget for URL
        self.url_entry = Entry(root, width=50, font=("Arial", 12))
        self.url_entry.pack(pady=10)
        self.placeholder_text = "Paste Image URL Here..."
        self.url_entry.insert(0, self.placeholder_text)
        self.url_entry.bind("<FocusIn>", self.clear_placeholder)
        self.url_entry.bind("<FocusOut>", self.add_placeholder)

        # Ensure Ctrl+A works
        self.url_entry.bind("<Control-a>", self.select_all)

        self.url_button = Button(root, text="Predict from URL", command=self.load_from_url, font=("Arial", 12))
        self.url_button.pack(pady=10)

        self.prediction_label = Label(root, text="", font=("Arial", 16, "bold"))
        self.prediction_label.pack(pady=20)

    def clear_placeholder(self, event):
        if self.url_entry.get() == self.placeholder_text:
            self.url_entry.delete(0, tk.END)

    def add_placeholder(self, event):
        if not self.url_entry.get():
            self.url_entry.insert(0, self.placeholder_text)

    def select_all(self, event):
        event.widget.select_range(0, tk.END)
        event.widget.icursor(tk.END)
        return "break"


    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.display_image(file_path)
            self.prediction_label.config(text="Processing...")
            self.root.update()  # Force update the GUI
            food_name = self.fv.get_food_name(file_path)
            self.prediction_label.config(text=f"Prediction: {food_name}")


    def load_from_url(self):
        url = self.url_entry.get()
        if url == self.placeholder_text or not url.strip():
            self.prediction_label.config(text="Please enter a valid URL.")
            return
        try:
            self.prediction_label.config(text="Processing...")
            self.root.update()  # Update UI immediately

            response = requests.get(url)
            response.raise_for_status()
            img_bytes = response.content

            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_resized = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img_resized)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk  # keep a reference

            img_for_model = tf.convert_to_tensor(img_bytes)
            food_name = self.fv.get_food_name(img_for_model, from_local=False)
            self.prediction_label.config(text=f"Prediction: {food_name}")
        except Exception as e:
            self.prediction_label.config(text="Image URL cannot be accessed!")

    def display_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk  # keep a reference


if __name__ == "__main__":
    root = tk.Tk()
    app = FoodVisionApp(root)
    root.mainloop()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Food Vision: Get the name of your food image!")
#     parser.add_argument('--filename', type=str, required=True, help='Path to the abstract file')

#     args = parser.parse_args()

#     # creating the food vision class
#     fv = FoodVision()

#     plt.imshow(tf.squeeze(fv.pre_process(args.filename))/255.0)
#     food_name = fv.get_food_name(args.filename)
#     plt.title(food_name, fontweight = 'bold', fontsize = 12)
#     plt.axis('off')
#     plt.show()



