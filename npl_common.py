"""
Common functions for project01
"""
import os
import pickle
from urllib import parse

#files for pickles
#TODO: use path
model_file = "./project01/npl_model.pickle"
model_age_file = "./project01/npl_model_age.pickle"
model_gender_file = "./project01/npl_model_gender.pickle"
cv_file = "./project01/npl_cv.pickle"
tf_file = "./project01/npl_tf.pickle"
enc_file = "./project01/npl_enc.pickle"


def urlParse(url):
    return re.split("^(https?\:)\/\/(([^:\/?#]*)(?:\:([0-9]+))?)([\/]{0,1}[^?#]*)(\?[^#]*|)(#.*|)$", url)

"""
def makeJSON(data)
    Парсим json и возращаем все url в виде строки
    data - json структура ["visits":{"url":"url,"timestamp":'time"}}
    
"""
def makeJSON(data):
    full_url_str = ""

    for url in data["visits"]:

        urls_part = parse.urlsplit(url['url'])
        domain = urls_part.netloc.split(".")
        for i in domain:
            full_url_str += " " + i

        path = urls_part.path.split("/")
        for i in path:
            s = i.split("-")
            for st in s:
                full_url_str += " " + st

            full_url_str = full_url_str.strip()

        """
        query_parts = parse.parse_qsl(urls_part.query)
        if len(query_parts) > 0:
            for i in query_parts:
                if not i[1].isnumeric():
                    full_url_str += " " + i[1]
                    
        """


    return full_url_str.strip()







def genderTransform(df):
    return df.replace({'F':0,'M':1})

def genderInvertTransform(df):
    return df.replace({0:'F',1:'M'})


#TODO: это все можно в один класс оформить
"""
def modelFileSave(filename, model)
    Сохраняем класс
"""
def modelFileSave(filename, model):
    with open('./' + filename, 'wb') as f:
        pickle.dump(model, f)
        os.chmod('./' + filename, 0o644)

#TODO: check file
"""
Загружаем pickle класс
"""
def modelFileLoad(filename):
    return pickle.load(open(filename, 'rb'))