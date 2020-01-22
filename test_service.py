# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:37:18 2020

@author: jwillert
"""

import requests
import pandas as pd

info = pd.read_csv('acct_info.csv')
hist = pd.read_csv('acct_hist.csv')

info_json = info.to_json()
hist_json = hist.to_json()

input_json = {'info':info_json,
              'hist':hist_json}

			
response = requests.post("http://127.0.0.1:5000/",json=input_json)
print(eval(response.text))
if response.status_code == 200:
    
    print("Test was a success")
    
    output = eval(response.text)
    
    test_result = output['result']
    predictions = eval(output['predictions'])
    
    print("The result form the test is: {}".format(test_result))
    print("The test returned the following predictions: ")
    print(predictions)
    
elif response.status_code == 500:
    
    print("Somethings broken...")