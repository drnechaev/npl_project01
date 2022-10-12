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
import npl_common as npl

#models

from sklearn import svm
from xgboost.sklearn import XGBClassifier



df = pd.read_csv("data_debug.txt", sep='\t')

df = df[(df.gender != '-') & (df.age != '-')]

d_learn = pd.DataFrame()
d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(npl.makeJSON)

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

#cls = svm.SVC()
cls = XGBClassifier();
print("Start teach classifier")
cls.fit(d_url,d_teach)

print("Teached... \nSaving...")


npl.modelFileSave(npl.model_file,cls)
npl.modelFileSave(npl.cv_file,cv)
npl.modelFileSave(npl.tf_file,tf)
npl.modelFileSave(npl.enc_file,enc)

print("End")


