# coding=utf-8
import sys
import os
import glob
import numpy as np
# import libraries
import sys
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import textract
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import random
import joblib
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, session

#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# Load your trained model
def import_model():
    crf=joblib.load("crf_model_Final_Clean2.joblib")
    return crf

# Load your trained model
model = import_model()
print('Model loaded. Check http://127.0.0.1:5000/')

def read_pdf(pdf_path):
    text = textract.process(pdf_path).decode('utf-8')
    return text

def pos_tags(document):
        sentences = nltk.sent_tokenize(document)
        print(len(sentences))
        sentences = [nltk.word_tokenize(sent) for sent in sentences]        
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        
        return sentences

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def get_skills(text,crf):

    text=text.lower()
    text = pos_tags(text)
    z=text
    text=[sent2features(s) for s in text]
    
    doc_pred =crf.predict(text)
# print(doc_pred)
    l=[]
    for i in range(len(z)):
      for j in range(len(z[i])):
        if doc_pred[i][j]=="B-skill" :
           if j<len(z[i])-2 and doc_pred[i][j+1]=="I-skill":
              if j<len(z[i])-3 and doc_pred[i][j+2]=="I-skill":
                skill=z[i][j][0]+' '+z[i][j+1][0]+' '+z[i][j+2][0]
                l.append(skill)
              else:
                skill=z[i][j][0]+' '+z[i][j+1][0]
                l.append(skill)              
           else:
              skill=z[i][j][0]
              l.append(skill)            
          

    print(list(set(l)))      
    skills=list(set(l))
    return skills


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')

def uploader_file():
   if request.method == 'POST':
      f = request.files['resume']
      # Save the file to ./uploads
      basepath = os.path.dirname(__file__)
      file_path = os.path.join(
      basepath, 'uploads', f.filename)
      f.save(file_path)
      return file_path


@app.route('/uploader', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file_path = uploader_file()
        text = read_pdf(file_path)
        #Make prediction and get skills
        prediction = get_skills(text,model) 
        return render_template('home.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug = True)
    
