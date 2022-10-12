"""
TESTING MODELS

"""

import pandas as pd
import numpy as np
import json
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


import npl_common as npl



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


#df = pd.read_csv("data.txt", sep='\t')
df = pd.read_csv("data_debug.txt", sep='\t')

df = df[(df.gender != '-') & (df.age != '-')]

d_learn = pd.DataFrame()
#TODO: d_url - df['user_json'].apply(json.load).apply(makeJSON)
d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(npl.makeJSON)

# Подгтовка данных, перевод в цифры
cv = CountVectorizer()
tf = TfidfTransformer()
d_url = cv.fit_transform(d_url)
d_url = tf.fit_transform(d_url)

# Провекра
#TODO: перенести на онехотэнкодер, можно пол так, а возраст по другому
d_teach = pd.DataFrame()
#d_teach['age_gender'] = df['gender'].map(str) + " " + df['age']
d_teach = df['gender'].map(str) + " " + df['age']
enc = LabelEncoder()
d_teach = enc.fit_transform(d_teach)


X_train, X_test, y_train, y_test = train_test_split(d_url,d_teach ,
                                   random_state=104,
                                   test_size=0.2,
                                   shuffle=True)

# models
from sklearn import svm   #67
from sklearn.multiclass import OneVsRestClassifier  #0.26 83%
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

cls = []

#cls.append( svm.SVC() )  #28,125 (full:29.136)
#cls.append( OneVsRestClassifier(LinearSVC(random_state=0)) )  #27,375 (full:26.79)
#cls.append(KNeighborsClassifier(n_neighbors=5)) #21,0 (full:
#cls.append(LogisticRegression()) #28,49 not work
#cls.append(GaussianNB()) #not work
#cls.append(MultinomialNB()) #28,0   (full:26.42)
#cls.append(SGDClassifier()) #27,625 (full:)
#cls.append(GradientBoostingClassifier()) #28,375 (full: 28.36)    long
cls.append(XGBClassifier()) #29,625  (full:29.219)


print("Start teach")

for idx, clsi in enumerate(cls):
    print ("Teach for cls{}".format(idx))
    clsi.fit(X_train, y_train)

print("Teached... \nSaving...")

# TESTING

columns=['gender','age','uid','user_json']
df = pd.read_csv("data_debug_test.txt", sep='\t', names=columns)

d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(npl.makeJSON)
d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

d_test = pd.DataFrame()


"""
for idx, item in enumerate(cls):
    print ("Teach for cls{}".format(idx))
    clsi.fit(d_url, d_teach)

result = cls.predict(d_url);

data_out = pd.DataFrame( enc.inverse_transform(result) )
data_out['uid'] = df['uid']
data_out["age"] = data_out[0].str.split(' ').str.get(0)
data_out["gender"] = data_out[0].str.split(' ').str.get(1)
data_out = data_out.drop(0,axis='columns')

data_out.sort_values(by='uid',axis = 0, ascending = True, inplace = True)

print(data_out)
"""

d_test = df['gender'].map(str) + " " + df['age']
d_test = enc.transform(d_test)

#cls.append(pickle.load(open(model_file, 'rb')))

for idx, clsi in enumerate(cls):
    accuracy = clsi.score(X_test, y_test)
    print ("Accurasy for cls{} named \"{}\" is {}".format(idx,type(clsi).__name__, accuracy * 100) )

"""
#end testing
"""