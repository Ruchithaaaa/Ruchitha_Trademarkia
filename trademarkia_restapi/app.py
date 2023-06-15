
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import re
import math
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import os
import sys
import numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")



@app.route("/", methods=['POST'])
def cancerPrediction():
    dataset_url = "https://raw.githubusercontent.com/Ruchithaaaa/classification-of-products-and-services/main/idmanual.json"
    with open('idmanual.json', 'r') as file:
        df = json.load(file)

    df.info()

    inputQuery1 = request.form['query1']
    # inputQuery2 = request.form['query2']
    # inputQuery3 = request.form['query3']
    # inputQuery4 = request.form['query4']
    # inputQuery5 = request.form['query5']
    
    
    texts = []  # List to store the text data
    labels = []
    
    for item in df:
        texts.append(item['description'])
        labels.append(item['class_id'])
        
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_labels)
    X_test = vectorizer.transform(test_labels)
    
    from sklearn.svm import LinearSVC
    model = LinearSVC()
    model.fit(X_train,train_labels)
    
    y_pred = model.predict(X_test)
    
    
    # report = classification_report(test_labels, y_pred)
    # print(report)


    # df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

    # train, test = train_test_split(df, test_size = 0.2)

    # features = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

    # train_X = train[features]
    # train_y=train.diagnosis
    
    # test_X= test[features]
    # test_y =test.diagnosis

    # model=RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # model.fit(train_X,train_y)

    # prediction=model.predict(test_X)
    # metrics.accuracy_score(prediction,test_y)
    data = [inputQuery1]
    # #print('data is: ')
    # #print(data)
    # #016.14, 74.00, 0.01968, 0.05914, 0.1619
    
    # # Create the pandas DataFrame 
    new_df = pd.DataFrame(df, columns = ['description'])
    single = model.predict(new_df)
    # probability = model.predict_proba(new_df)[:,1]
    print(single)
    # if single==1:
    #     output = "The patient is diagnosed with Breast Cancer"
    #     output1 = "Confidence: {}".format(probability*100)
    # else:
    #     output = "The patient is not diagnosed with Breast Cancer"
    #     output1 = ""
    output1="abc"
    output="class: {}".format(single)
    
    return render_template('home.html', output1=output, query1 = request.form['query1'])
    
app.run()

