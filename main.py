#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:27:43 2022

@author: nicola
"""
import pandas as pd
import json
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#models
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

"""
    return 
    [1] protocol
    [2] url
    [3] url
    [5] path
    [6] gets values
    [7]
    [8] hash
"""
def urlParse(url):
    return re.split("^(https?\:)\/\/(([^:\/?#]*)(?:\:([0-9]+))?)([\/]{0,1}[^?#]*)(\?[^#]*|)(#.*|)$",   url)
 
    
def makeJSON(data):
    full_url_str = ""
    for url in data["visits"]:
       urls_part = urlParse((url["url"]))
       full_url_str += " " + urls_part[3] + " " +urls_part[5]
       
    return full_url_str.strip()
       
        
       
df = pd.read_csv("data_debug.txt", sep='\t')

d_learn = pd.DataFrame()
d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(makeJSON)

#Подгтовка данных, перевод в цифры
cv = CountVectorizer()
tf = TfidfTransformer()
d_url = cv.fit_transform(d_url)
d_url = tf.fit_transform(d_url)


d_teach = pd.DataFrame()
d_teach['gender'] = df['gender']
d_teach['age'] = df['age']
enc = LabelEncoder();
d_t = enc.fit_transform(d_teach['gender'])
#print(d_teach)

cls = MultinomialNB()
cls2 = KNeighborsClassifier(n_neighbors=5)
cls.fit(d_url, d_t)
cls2.fit(d_url,d_t)

#testing
df = pd.read_csv("data_debug_test.txt", sep='\t')

d_learn = pd.DataFrame()
d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(makeJSON)
d_t = enc.transform(df['gender'])


print ( d_url )
d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

print (cls.predict(d_url))
print (cls2.predict(d_url))
print(d_t)

accuracy = cls.score(d_url, d_t)
print("Accuracy = {}%".format(accuracy * 100))

accuracy = cls2.score(d_url, d_t)
print("Accuracy = {}%".format(accuracy * 100))
