import pandas as pd
import json
import re
from sklearn.preprocessing import LabelEncoder
import pickle
import os
# models

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
    return re.split("^(https?\:)\/\/(([^:\/?#]*)(?:\:([0-9]+))?)([\/]{0,1}[^?#]*)(\?[^#]*|)(#.*|)$", url)


def makeJSON(data):
    full_url_str = ""
    for url in data["visits"]:
        urls_part = urlParse((url["url"]))
        full_url_str += " " + urls_part[3] + urls_part[5]

    return full_url_str.strip()

print("=========================\nSTART TESTING\n=========================")
"""
#testing
"""

cv = pickle.load(open("./project01/npl_cv.pickle", 'rb'))
tf = pickle.load(open("./project01/npl_tf.pickle", 'rb'))

df = pd.read_csv("data_debug_test.txt", sep='\t')
df = df[(df.gender != '-') & (df.age != '-')]

d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(makeJSON)
d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

enc = pickle.load(open("./project01/npl_enc.pickle","rb"))
d_test = pd.DataFrame()

d_test['age_gender'] = df['gender'].map(str) + " " + df['age']
d_test = enc.transform(d_test)


"""
#end testing
"""
cls = pickle.load(open("./project01/npl_model.pickle", 'rb'))

result = cls.predict(d_url);

data_out = pd.DataFrame( enc.inverse_transform(result) )

data_out['uid'] = df['uid']
data_out["age"] = data_out[0].str.split(' ').str.get(0)
data_out["gender"] = data_out[0].str.split(' ').str.get(1)
data_out = data_out.drop(0,axis='columns')

data_out.sort_values(by='uid',axis = 0, ascending = True, inplace = True)

print(data_out)

accuracy = cls.score(d_url, d_test)
print("Accuracy cls = {}%".format(accuracy * 100))