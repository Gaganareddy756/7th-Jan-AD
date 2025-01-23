#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install SpeechRecognition pyttsx3 pyaudio


# In[2]:


import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def main():
    speak("Hello! This is a simple chatbot from Malla Reddy University.")
    speak("You can say hello, ask my name, or say goodbye.")
    while True:
        command = input("You: ").lower()
        if "hello" in command:
            speak("Hi there! Welcome to Malla Reddy University.")
        elif "what's your name" in command or "what is your name" in command:
            speak("My name is Simple Bot from Malla Reddy University.")
        elif "goodbye" in command:
            speak("Goodbye! Have a great day at Malla Reddy University.")
            break
        else:
            speak("I don't understand that. Please try again.")

if __name__ == "__main__":
    main()


# In[ ]:


"D:\AD Classes\Details.csv"


# In[8]:


import pandas as pd
df=pd.read_csv(r"D:\AD Classes\Details.csv")
print(df.head(7))
print(df.head())


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
df=pd.read_csv(r"D:\AD Classes\Details.csv")
print(df.columns)
plt.scatter(df['Age'],df['Purchased'],color='b',label='Purchased')
plt.xlabel('Age')
plt.ylabel('Purchased')
plt.legend()
plt.show()


# In[6]:


import pyttsx3
import speech_recognition as sr

engine = pyttsx3.init()

def speak(text):
    """Make the assistant speak."""
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    """Capture input from the microphone and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand that. Please try again.")
            return ""
        except sr.RequestError:
            speak("Network error. Please check your internet connection.")
            return ""

def main():
    speak("Hello, I am a simple bot from PK Karthikeya.")
    speak("You can say hello, ask my name, or say goodbye.")
    
    while True:
        speak("Please give your command.")
        command = get_voice_input()  # For voice input
        
        
        if "hello" in command:
            speak("Hi there! Welcome to Malla Reddy College.")
        elif "what's your name" in command or "what is your name" in command:
            speak("My name is Simple Bot from Malla Reddy College.")
        elif "goodbye" in command:
            speak("Goodbye! Have a great day at MRU.")
            break
        elif command:
            speak("I didn't understand that. Please try again.")

if __name__ == "__main__":
    main()


# In[9]:


import nltk
#nltk.download('punkt') .This is not mandatory as the punkt is already downloaded
from nltk.tokenize import word_tokenize
text = "This is an example"
tokens = word_tokenize(text)
print(tokens)


# In[ ]:


from flask import Flask
app= Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'
if __name__ == '__main__' :
    app.run(debug=True)


# In[ ]:


from flask import Flask, render_template
app=Flask(__name__)
@app.router("/")
def home():
    return render_template("index.html")
if __name__=="__main__":
    app.run(debug=True)


# In[3]:


from flask import  Flask, render_template, request
from transformers import pipeline

app= Flask(__name__)

generator = pipeline("text-generation",model="gpt2-large")

@app.route("/")

def home():
    return render_template("index.html")
@app.route("/generate", methods=["GET", "POST"])

def generate():
    if request.method =="GET":
        return home()
    prompt=request.form["prompt"]
    if not prompt:
        return "Please enter a prompt"
    
max_length=int(request.form.get("max_length", 100))
temperature = float(request.form.get("temperature",0.7))

genetrated_text=generator(prompt, max_length=max_length,num_return_sequences=1)
final_result=generated_text[0]["generated_text"]
return render_template("index.html", prompt=prompt, result=final_result, max_length=max_length, temperature=temperature)
if __name__ == '__main__':
    app.run(debug=True)


# In[3]:


get_ipython().system('pip install transformers')


# In[11]:


import pandas as pd
dataset = pd.read_csv('D:/AD Classes/tweets.csv', encoding='ISO-8859-1')
dataset.head()


# In[12]:


import re
def clean_text(text):
    text = re.sub(r'RT','',text)
    
    text = re.sub(r'&amp;','',text)
    
    text  = re.sub(r'[?!.;:,#@-]','',text)
    
    text = re.sub(r'[^a-zA-Z\']',' ',text)
    
    text = re.sub(r'[^\x00-\x7F]+',' ',text)
    
    text = text.lower()
    return text


# In[10]:


dataset['clean_text']=dataset.text.apply(
    lambda x:clean_text(x))
dataset['clean_text'].head()


# In[15]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
print(stop)


# In[14]:


import nltk===
nltk.download('stopwords')


# In[2]:


get_ipython().system(' python -m spacy download en_core_web_sm')


# In[3]:


pip install spacy


# In[6]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[ ]:


get_ipython().system('conda update conda')
get_ipython().system('conda update anaconda')


# In[ ]:




