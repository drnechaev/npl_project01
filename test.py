import pandas as pd
import json
import re
import pickle

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
columns=['gender','age','uid','user_json']
df = pd.read_csv("data_debug.txt", sep='\t') #, names=columns)


def len_json(data):
    return len(data["visits"])

d_url = df["user_json"]
d_url = d_url.apply(json.loads)
df['json_len'] = d_url.apply(len_json)
#d_url = d_url.apply(makeJSON)

print( df.groupby(["json_len"])['json_len'].count())

exit(1)
cv = pickle.load(open("./project01/npl_cv.pickle", 'rb'))
tf = pickle.load(open("./project01/npl_tf.pickle", 'rb'))

d_url = cv.transform(d_url)
d_url = tf.transform(d_url)

enc = pickle.load(open("./project01/npl_enc.pickle","rb"))
d_test = pd.DataFrame()



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

d_test['age_gender'] = df['gender'].map(str) + " " + df['age']
d_test = enc.transform(d_test)

accuracy = cls.score(d_url, d_test)
print("Accuracy cls = {}%".format(accuracy * 100))