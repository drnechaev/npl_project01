"""
TESTING MODELS

"""

import pandas as pd
import numpy as np
import json
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


import npl_common as npl


#df = pd.read_csv("data.txt", sep='\t')
df = pd.read_csv("data_debug_test.txt", sep='\t')

df = df[(df.gender != '-') & (df.age != '-')]

d_url = df["user_json"].apply(json.loads).apply(npl.makeJSON)


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
#d_teach = df['age']
enc = LabelEncoder()
d_teach = enc.fit_transform(d_teach)


X_train, X_test, y_train, y_test = train_test_split(d_url,d_teach ,
                                   random_state=104,
                                   test_size=0.05,
                                   shuffle=True)

# models
from sklearn import svm   #67
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier  #0.26 83%
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

cls = []

#cls.append( svm.SVC() )  #28,125 (full:29.136)
#cls.append( OneVsRestClassifier(LinearSVC(random_state=0)) )  #27,375 (full:26.79)
#cls.append(KNeighborsClassifier(n_neighbors=5)) #21,0 (full:
#cls.append(LogisticRegression()) #28,49 not work
#cls.append(GaussianNB()) #not work
#cls.append(MultinomialNB()) #28,0   (full:26.42)
#cls.append(SGDClassifier()) #27,625 (full:)
#cls.append(GradientBoostingClassifier()) #28,375 (full: 28.36)    long
#cls.append(XGBClassifier()) #29,625  (full:29.219)
#cls.append(CatBoostClassifier(iterations=150, random_seed=45, loss_function='MultiClass')) #29.579
# random_seed=45 (30,0624)
#cls.append( OneVsRestClassifier(estimator=CatBoostClassifier(iterations=150, random_seed=43)) ) #29,27
cls.append( OneVsOneClassifier(estimator=CatBoostClassifier(iterations=150, random_seed=45)) ) #


print("Start teach")

for idx, clsi in enumerate(cls):
    print ("Teach for cls{}".format(idx))
    clsi.fit(X_train, y_train)

print("Teached... \nSaving...")

# TESTING

for idx, clsi in enumerate(cls):
    accuracy = clsi.score(X_test, y_test)
    print ("Accurasy for cls{} named \"{}\" is {}".format(idx,type(clsi).__name__, accuracy * 100) )

   # result = clsi.predict(d_url);

"""
    data_out = pd.DataFrame(enc.inverse_transform(result))
    data_out['uid'] = df['uid']
    data_out["age"] = data_out[0].str.split('_').str.get(0)
    data_out["gender"] = data_out[0].str.split('_').str.get(1)
    data_out = data_out.drop(0, axis='columns')

    data_out.sort_values(by='uid', axis=0, ascending=True, inplace=True)

    print(data_out)
"""
"""
#end testing
"""