import numpy as np
import json
from voice_recognition_ml import *
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd


with open('train_mfcc_data.json', 'r', encoding='utf-8') as f: 
    data = json.load(f) 
z = np.array(data['mapping']) # get the mapping from index to corresponding names

window = tk.Tk() 
window.geometry("300x300") # window size
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

    if model == 'mfcc':        
        loaded_model = pickle.load(open('VoiceRecognitionModel.pkl', 'rb')) # load the model from disk 
    else:
        loaded_model = pickle.load(open('VoiceRecognitionModel.pkl', 'rb')) # load the model from disk

    filename = fd.askopenfilename() # filename of chosen file from file picker
    pred, _ = new_voice_predict(filename, loaded_model, z) 

    label2 = Label(window, text='Predicted voice:\n'+str(pred), width=25, font=('times',10)) # add predicted name as label
    label2.grid(row=5, column=1) # label position
    print(pred)

window.mainloop() # run the gui