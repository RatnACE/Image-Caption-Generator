 #!/usr/bin/env python
# coding: utf-8

# # Import Modules
# 

# In[1]:


import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
#import matplotlib.pyplot as plt
import pickle
import numpy as np


# # Loading Model

# In[2]:


#load the trained model to classify sign
model = load_model('image caption gen.h5')


# # Initialise GUI

# In[3]:


#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Image caption Generator')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)


# # variables and loading trained data

# In[4]:


with open(('tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

max_length=35


# # Caption Data generation functions

# In[5]:


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text
def generator(file_path):
    model3 = VGG16()
    # restructure the model
    model3 = Model(inputs=model3.inputs, outputs=model3.layers[-2].output)
    key="alpha"
    #print(model3.summary())

    featured = {}
    
    # load the image
    img_path = file_path
    image1 = Image.open(img_path)
    
    
    ###
    # load the image from file
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = model3.predict(image, verbose=0)
        # get image ID
    featured[key] = feature
    #print(featured[key])
      
    y_pred = predict_caption(model, featured[key], tokenizer, max_length)
    predict=y_pred.replace("startseq","")
    predict=predict.replace("endseq",".")
    label.configure(foreground='#011638', text=predict)
    #print('--------------------Predicted--------------------')
    #print(y_pred)
    #plt.imshow(image1)


# # Buttons and GUI finalisation

# In[6]:



def show_classify_button(file_path):
  classify_b=Button(top,text="Generate Caption",command=lambda: generator(file_path),padx=10,pady=5)
  classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
  classify_b.place(relx=0.79,rely=0.46)

def upload_image():
  try:
      file_path=filedialog.askopenfilename()
      uploaded=Image.open(file_path)
      uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
      im=ImageTk.PhotoImage(uploaded)

      sign_image.configure(image=im)
      sign_image.image=im
      label.configure(text='')
      show_classify_button(file_path)
  except:
      pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Caption Generator.",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()


# In[ ]:




