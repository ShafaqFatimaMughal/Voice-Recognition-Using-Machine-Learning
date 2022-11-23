import numpy as np
import json
from voice_recognition_ml import *
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

with open(os.path.join(script_dir, 'Resources/train_mfcc_data.json'), 'r', encoding='utf-8') as f: 
    data = json.load(f) 
z = np.array(data['mapping']) # get the mapping from index to corresponding names

window = tk.Tk() 
window.geometry("300x200") # window size
window.title("Voice Recognition")

label1 = Label(window, text='Upload WAV File', width=25, font=('times',15,'bold')) # Main label
label1.grid(row=1, column=1) # label position

var = StringVar() # used to get the 'value' property of a tkinter.Radiobutton
rb1 = Radiobutton(text='MFCC Model', variable=var, value="mfcc")
rb1.grid(row=2, column=1) # radiobutton position
rb2 = Radiobutton(text='STFT Model', variable=var, value="stft")
rb2.grid(row=3, column=1) # radiobutton position

button1 = Button(window, text='File Picker', width=20, command=lambda:upload_file()) # button
button1.grid(row=4, column=1) # button position

def upload_file():
    model = var.get() # value of the radio button selected

    label2 = Label(window, text='Predicting Voice...', width=25, font=('times',10)) # add predicted name as label
    label2.grid(row=5, column=1) # label position

    if model == 'mfcc':  
        filepath = os.path.join(script_dir, 'Resources/mfcc_VoiceRecognitionModel.pkl') 
        loaded_model = pickle.load(open(filepath, 'rb')) # load the model from disk 
    else:
        filepath = os.path.join(script_dir, 'Resources/stft_VoiceRecognitionModel.pkl') 
        loaded_model = pickle.load(open(filepath, 'rb')) # load the model from disk

    filename = fd.askopenfilename() # filename of chosen file from file picker
    pred, _ = new_voice_predict(filename, loaded_model, z, model) 

    label2.config(text='Predicted voice:\n'+str(pred)) # add predicted name as label
    print(pred)

window.mainloop() # run the gui