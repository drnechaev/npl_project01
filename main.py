#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:27:43 2022

@author: nicola
"""

import pandas as pd
import json
import re
from urllib.parse import urlparse
from urllib.request import urlretrieve, unquote
from datetime import datetime as dt


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
    return re.split("^(https?\:)\/\/(([^:\/?#]*)(?:\:([0-9]+))?)([\/]{0,1}[^?#]*)(\?[^#]*|)(#.*|)$",   url)
 
    
def makeJSON(data):
    full_url_str = ""
    for url in data["visits"]:
       urls_part = urlParse((url["url"]))
       full_url_str += " " + urls_part[3] + " " +urls_part[5]
       
    return full_url_str.strip()
       
        
       
df = pd.read_csv("data.txt", sep='\t')

d_test = pd.DataFrame()

d_test["urls"]= df["user_json"]
#df["user_json"] = df["user_json"].apply(json.loads)
d_test["urls"] = d_test["urls"].apply(json.loads)
d_test["urls"] = d_test["urls"].apply(makeJSON)

print (d_test.loc[0].urls)