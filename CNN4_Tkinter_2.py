# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:33:52 2018

@author: rutwi
"""

# CNN4_Tkinter_2

#Loading The Model

from keras.models import load_model
classifier = load_model('My_CNN_Model.h5')


# **************** TKINTER PART ***********************************************

#Taking Image from User (TK File Dialog)

from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import PIL

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


root = Tk()


def select_image2(self):
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    path = root.filename
    print (root.filename)
    display_image(path)
    new_img = load_image(path)
    
    pred = classifier.predict(new_img)
    
    #Printing the Prediction
    if (pred <= 0.5):
        text_label['text'] = "Its a Cat..."
        text_label['fg'] = 'red'
        print("Its a Cat !!")
    else:
        text_label['text'] = "Its a Dog! Yeaahhh !!!! "
        text_label['fg'] = 'blue'
        print("Its a Dog! Yeahh!!")
    
#Displaying the Image on Tkinter Window
def display_image(path):
    
    #Image Resizing For Display on Tkinter Window
    basewidth = 300
    img = Image.open(path)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save('resized_image.jpg')
    
    #Displaying the Image on Tkinter Window as a Label
    photo = ImageTk.PhotoImage(file = "resized_image.jpg")
    
    image_label['image'] = photo
    
    image_label.image = photo
    
       
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor  


Button2 = Button(root, text ="Select an Image" )
Button2.bind("<Button-1>", select_image2)
Button2.pack()

Button(root, text="Quit New", command=root.destroy).pack()

text_label = Label(root, font=("Helvetica", 16)) 
text_label.pack()

image_label = Label(root)
image_label.pack()

#Make window appear constantly
root.mainloop()
