#!/usr/bin/env python3

import pandas as pd
import json
import re
import pickle
import sys

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


cv = pickle.load(open("./project01/npl_cv.pickle", 'rb'))
tf = pickle.load(open("./project01/npl_tf.pickle", 'rb'))

#df = pd.read_csv("data_debug_test.txt", sep='\t')
columns=['gender','age','uid','user_json']

df = pd.read_table(
    sys.stdin,
    sep='\t',
    header=None,
    names=columns
)


d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(makeJSON)
d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

enc = pickle.load(open("./project01/npl_enc.pickle","rb"))

cls = pickle.load(open("./project01/npl_model.pickle", 'rb'))
result = cls.predict(d_url);

data_out = pd.DataFrame( enc.inverse_transform(result) )

data_out['uid'] = df['uid']
data_out["age"] = data_out[0].str.split(' ').str.get(0)
data_out["gender"] = data_out[0].str.split(' ').str.get(1)
data_out = data_out.drop(0,axis='columns')

data_out.sort_values(by='uid',axis = 0, ascending = True, inplace = True)

sys.stdout.write(data_out.to_json(orient='records'))
