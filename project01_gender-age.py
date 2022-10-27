#!/opt/userenvs/nikolay.nechaev/prj01/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import npl_common as npl
import sys

cv = npl.modelFileLoad(npl.cv_file)
tf = npl.modelFileLoad(npl.tf_file)


columns=['gender','age','uid','user_json']
#df = pd.read_csv("data_debug_test.txt", sep='\t',names=columns)


df = pd.read_table(
    sys.stdin,
    sep='\t',
    header=None,
    names=columns
)


df = df[(df.gender == '-') & (df.age == '-')].reset_index()

d_url = df["user_json"].apply(json.loads).apply(npl.makeJSON)
d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

enc = npl.modelFileLoad(npl.enc_file)

cls_age = npl.modelFileLoad(npl.model_age_file)
cls_gender = npl.modelFileLoad(npl.model_gender_file)

p_age = cls_age.predict(d_url)
p_gender = cls_gender.predict(d_url)

data_out = pd.DataFrame()

data_out['uid'] = df['uid']
data_out["gender"] = p_gender
data_out["gender"] = npl.genderInvertTransform(data_out["gender"])
data_out["age"] = enc.inverse_transform(p_age)


data_out.sort_values(by='uid',axis = 0, ascending = True, inplace = True)

sys.stdout.write(data_out.to_json(orient='records'))
