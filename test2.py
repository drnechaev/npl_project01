"""
TESTING MODELS

"""

import pandas as pd
import numpy as np
import json
import re
import os.path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


import npl_common as npl


pd.set_option("mode.chained_assignment", None)

df = pd.read_csv("data.txt", sep='\t')
#df = pd.read_csv("data_debug_test.txt", sep='\t')

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
#d_teach = df['gender'].map(str) + " " + df['age']
#d_teach = df['age']
enc = LabelEncoder()
d_teach = df[['gender','age']]

d_teach['gender'] = d_teach['gender'].replace({'F':0,'M':1})
d_teach['age'] = enc.fit_transform(d_teach['age'])



X_train, X_test, y_train, y_test = train_test_split(d_url,d_teach ,
                                   test_size=0.05,
                                   shuffle=True)


# models
from sklearn import svm   #67
from xgboost.sklearn import XGBClassifier
"""
Accurasy for age cls named "XGBClassifier" is 42.94410625345877
predicted values:
25-34    1491
35-44     213
45-54      64
18-24      25
>=55       14
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from catboost import CatBoostClassifier  #45
"""
Accurasy for age cls named "CatBoostClassifier" is 43.99557277255119
predicted values:
25-34    1489
35-44     244
45-54      38
18-24      25
>=55       11
"""


#cls_age = MultinomialNB()
#cls_age = CatBoostClassifier(iterations=150, random_seed=45, loss_function='MultiClass')
cls_age = XGBClassifier()


print("Start teach")
if os.path.exists("gender_model.pickle"):
    cls_gender = npl.modelFileLoad("gender_model.pickle")
else:
    cls_gender = CatBoostClassifier(iterations=150, random_seed=45)
    cls_gender.fit(X_train,y_train['gender'])
    npl.modelFileSave("gender_model.pickle",cls_gender)


cls_age.fit(X_train,y_train['age'])


print("Teached... \nSaving...")

# TESTING
out = pd.DataFrame()
p_age = cls_age.predict(X_test)


p_gender = cls_gender.predict(X_test)
accuracy = cls_gender.score(X_test, y_test['gender'])
print ("Accurasy for gender cls named \"{}\" is {}".format(type(cls_gender).__name__, accuracy * 100) )
out['gender'] = p_gender
out['gender'] = out['gender'].replace({0:'F',1:'M'})


accuracy = cls_age.score(X_test, y_test['age'])
print ("Accurasy for age cls named \"{}\" is {}".format(type(cls_age).__name__, accuracy * 100) )



out['age'] = enc.inverse_transform(p_age)




#y_test['gender_p'] = p_gender
#y_test['gender_p'] =y_test['gender_p'].replace({0:'F',1:'M'})

#y_test['age_p2'] = p_age
#y_test['gender_p2'] = p_gender

print("predicted values:")
print(out['age'].value_counts())
y_test['age'] = enc.inverse_transform(y_test['age'])
print("real values:")
print(y_test['age'].value_counts())



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