#!/opt/userenvs/nikolay.nechaev/prj01/bin/python3
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
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost.sklearn import XGBClassifier


pd.set_option("mode.chained_assignment", None)
df = pd.read_csv("data.txt", sep='\t')

df = df[(df.gender != '-') & (df.age != '-')]

d_learn = pd.DataFrame()
d_url = df["user_json"].apply(json.loads).apply(npl.makeJSON)

#Подгтовка данных, перевод в цифры
cv = CountVectorizer()
tf = TfidfTransformer()
d_url = cv.fit_transform(d_url)
d_url = tf.fit_transform(d_url)

#Провекра
#d_teach = pd.DataFrame()
#d_teach['age_gender'] = df['gender'].map(str) + " " + df['age']
d_teach = df[['gender','age']]
enc = LabelEncoder()
d_teach['age'] = enc.fit_transform(d_teach['age'])
d_teach['gender'] = npl.genderTransform(d_teach['gender'])


cls_age = SVC()
#cls_age =  XGBClassifier()
cls_gender = XGBClassifier()
#cls_gender = CatBoostClassifier(iterations=150, random_seed=43);


print("Start teach age classifier")
cls_age.fit(d_url,d_teach['age'])
print("Teached...\nStart teach gender classifier")
cls_gender.fit(d_url,d_teach['gender'])

print("Teached... \nSaving...")


npl.modelFileSave(npl.model_age_file,cls_age)
npl.modelFileSave(npl.model_gender_file,cls_gender)
npl.modelFileSave(npl.cv_file,cv)
npl.modelFileSave(npl.tf_file,tf)
npl.modelFileSave(npl.enc_file,enc)

print("End")


