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
import pickle
import os


model_file = "npl_model.pickle"
cv_file = "npl_cv.pickle"
tf_file = "npl_tf.pickle"
enc_file = "npl_enc.pickle"

#models

from sklearn import svm

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
        full_url_str += " " + urls_part[3]  + urls_part[5]
       
    return full_url_str.strip()

def modelFileSave(filename, model):
    with open('./' + filename, 'wb') as f:
        pickle.dump(model, f)
        os.chmod('./' + filename, 0o644)


df = pd.read_csv("data.txt", sep='\t')

df = df[(df.gender != '-') & (df.age != '-')]

d_learn = pd.DataFrame()
d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(makeJSON)

#Подгтовка данных, перевод в цифры
cv = CountVectorizer()
tf = TfidfTransformer()
d_url = cv.fit_transform(d_url)
d_url = tf.fit_transform(d_url)

#Провекра
d_teach = pd.DataFrame()
d_teach['age_gender'] = df['gender'].map(str) + " " + df['age']
enc = LabelEncoder()
d_teach = enc.fit_transform(d_teach['age_gender'])

cls = svm.SVC()
print("Start teach svc")
cls.fit(d_url,d_teach)

print("Teached... \nSaving...")


modelFileSave(model_file,cls)
modelFileSave(cv_file,cv)
modelFileSave(tf_file,tf)
modelFileSave(enc_file,enc)

print("End")


