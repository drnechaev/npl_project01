#!/usr/bin/env python3

import pandas as pd
import json
import re
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



cv = npl.modelFileLoad(npl.cv_file) #pickle.load(open(cv_file, 'rb'))
tf = npl.modelFileLoad(npl.tf_file) #pickle.load(open(tf_file, 'rb'))

#df = pd.read_csv("data_debug_test.txt", sep='\t')
columns=['gender','age','uid','user_json']

df = pd.read_table(
    sys.stdin,
    sep='\t',
    header=None,
    names=columns
)

df = df[(df.gender == '-') & (df.age == '-')].reset_index()

d_url = df["user_json"]
d_url = d_url.apply(json.loads)
d_url = d_url.apply(npl.makeJSON)
d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

enc = npl.modelFileLoad(npl.enc_file)

cls = npl.modelFileLoad(npl.model_file)
result = cls.predict(d_url);

data_out = pd.DataFrame( enc.inverse_transform(result) )

data_out['uid'] = df['uid']
data_out["gender"] = data_out[0].str.split(' ').str.get(0)
data_out["age"] = data_out[0].str.split(' ').str.get(1)
data_out = data_out.drop(0,axis='columns')

data_out.sort_values(by='uid',axis = 0, ascending = True, inplace = True)

sys.stdout.write(data_out.to_json(orient='records'))
