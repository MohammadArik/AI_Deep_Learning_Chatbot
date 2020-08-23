"""AI Chatbot by Mohammad Arik"""

# ----------------------------------------------------------------------------------------------------------------------
# importing necessary modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import json
import pickle
import pyttsx3
import tkinter as tk
from tkinter import ttk
import random
import webbrowser
import speech_recognition as sr

# ----------------------------------------------------------------------------------------------------------------------
# backgrounds of the interface
backgrounds = ["./images/backgrounds/1.png", "./images/backgrounds/2.png", "./images/backgrounds/3.png",
               "./images/backgrounds/4.png", "./images/backgrounds/5.png", "./images/backgrounds/6.png",
               "./images/backgrounds/7.png", "./images/backgrounds/8.png"]

# setting speech recognition classes
r1 = sr.Recognizer()
r2 = sr.Recognizer()

# setting class for the word stemmer
stemmer = LancasterStemmer()

# initializing pyttsx for speaking as the bot
pyttsx3.init("sapi5")
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', 200)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# ----------------------------------------------------------------------------------------------------------------------
# trying to load pre-processed data set
try:
    with open("data.pickle", "rb") as f:
        words, labels, train, output = pickle.load(f)

# if the processed data isn't found then process the data
except FileNotFoundError:
    with open('intents.json') as file:
        data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            words.extend(nltk.word_tokenize(pattern))
            docs_x.append(nltk.word_tokenize(pattern))
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?" and "!"]
    words = sorted(list(set(words)))
    labels.sort()

    train = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        train.append(bag)
        output.append(output_row)

    train = np.array(train)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, train, output), f)

# ----------------------------------------------------------------------------------------------------------------------
# modeling the neural network
tf.keras.backend.clear_session()
model = Sequential()
model.add(Dense(8, input_dim=len(train[0]), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ----------------------------------------------------------------------------------------------------------------------
# trying to load the model
try:
    model = load_model("chatbot.h5")

# train the model if not found
except OSError:
    model.fit(train, output, epochs=500, batch_size=8)
    model.save("chatbot.h5")


# ----------------------------------------------------------------------------------------------------------------------
# funtion to process the input
def bag_of_words(s):
    bag_in = [0 for _ in range(len(words))]
    ret = []
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for t in s_words:
        for i, w in enumerate(words):
            if w == t:
                bag_in[i] = 1

    ret.append(bag_in)
    ret = np.array(ret)
    return ret


# ----------------------------------------------------------------------------------------------------------------------
# loading the intents.json file
with open('intents.json') as file:
    data = json.load(file)


# ----------------------------------------------------------------------------------------------------------------------
# function for going to my Youtube channel
def youtube():
    webbrowser.open('https://www.youtube.com/channel/UCfdIVBdrbA3D7fUyBtme6xg?view_as=subscriber')


# ----------------------------------------------------------------------------------------------------------------------
# function for going to my Facebook Page
def facebook():
    webbrowser.open('https://www.facebook.com/khondaker.arik')


# ----------------------------------------------------------------------------------------------------------------------
# function for going to my instagram account
def instagram():
    webbrowser.open('https://www.instagram.com/cooper_arik.62/?hl=en')


# ----------------------------------------------------------------------------------------------------------------------
# calling the class for tkinter window
root = tk.Tk()


# ----------------------------------------------------------------------------------------------------------------------
# initializing the window
def main_window(main):
    def text(txt):
        for i in range(len(txt)):
            entry.delete(0, last=None)
        tk.Label(main_frame, text=txt, fg="#e1e1e1", bg="#1e1e1e", padx=25).pack(anchor='e')

        results = model.predict(bag_of_words(txt))
        results_index = np.argmax(results)
        tag = labels[int(results_index)]

        for query in data["intents"]:
            if query["tag"] == tag:
                responses = query["responses"]

        out = random.choice(responses)
        tk.Label(main_frame, text=out, bg="#e1e1e1", fg="#1e1e1e", padx=25).pack(anchor='w')
        return speak(out)

    def speak(sentence):
        engine.say(sentence)
        engine.runAndWait()

    def listen():
        with sr.Microphone() as source:
            audio = r1.listen(source)
            
        try:
            e = r2.recognize_google(audio)
            entry.insert(0, e)
            
        except sr.UnknownValueError:
            engine.say("Sorry I didn't understand")
            tk.Label(main_frame, text="Sorry I didn't understand").pack()

        except sr.RequestError:
            engine.say("Please connect to the Internet.")
            engine.runAndWait()

    main.geometry('400x700')
    main.title('A.I. Deep Learning Chatbot')
    main.iconbitmap('./images/icon.ico')

    head = tk.PhotoImage(file='./images/head.png')
    back = tk.PhotoImage(file=random.choice(backgrounds))
    fb = tk.PhotoImage(file='./images/fb.png')
    insta = tk.PhotoImage(file='./images/insta.png')
    tube = tk.PhotoImage(file='./images/youtube.png')
    speaker = tk.PhotoImage(file='./images/speak.png')

    tk.Label(main, image=head, bd=0).place(relx=0, y=0)
    tk.Label(main, image=back, bd=0).place(relx=0, y=140)
    tk.Label(main, bg='#343434', padx=1400, pady=70).place(x=0, rely=0.85)
    tk.Label(main, bg='#636363', padx=1400, pady=10).place(x=0, rely=0.8)
    tk.Button(main, image=fb, bg='#343434', bd=0, relief='groove', command=lambda: facebook()).place(x=15, rely=0.9)
    tk.Button(main, image=insta, bg='#343434', bd=0, relief='groove', command=lambda: instagram()).place(x=80, rely=0.9)
    tk.Button(main, image=tube, bg='#343434', bd=0, relief='groove', command=lambda: youtube()).place(x=145, rely=0.9)

    entry = tk.Entry(main, cursor='arrow', width=22, bg='#e1e1e1', font='none 16')
    entry.place(x=10, rely=0.805)

    tk.Button(main, text='Send', padx=15, pady=5, bg='#363636', bd=0, relief='groove',
              command=lambda: text(entry.get())).place(x=280, rely=0.805)
    
    tk.Button(main, image=speaker, padx=10, bd=0, bg='#363636', relief='groove',
              command=lambda: listen()).place(x=350, rely=0.805)

    container = ttk.Frame(main)
    canvas = tk.Canvas(container, height=400, width=380)
    scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
    main_frame = ttk.Frame(canvas)
    main_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=main_frame)
    canvas.configure(yscrollcommand=scrollbar.set)
    container.place(x=0, y=150)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    tk.Label(main_frame, text='', padx=200, bg='#000').pack()

    main.mainloop()


main_window(root)
