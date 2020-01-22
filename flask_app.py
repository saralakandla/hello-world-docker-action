# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:30:01 2020

@author: jwillert
"""

from flask import Flask, jsonify, request
import pandas as pd
import ep_usage_test_v2

app = Flask(__name__)

PLOT_ON = False

@app.route("/",methods=['POST','GET'])
def index():
    
    if (request.method == 'POST'):
        print('Post Method Called')
        input_data = dict(request.get_json())
        
        account_info = pd.read_json(input_data['info'])
        account_hist = pd.read_json(input_data['hist'])
        
        test_result, predictions = ep_usage_test_v2.ep_usage_test(account_hist,
                                                                  account_info,
                                                                  PLOT_ON)
        output = jsonify({'result':test_result,
                         'predictions':str(predictions.tolist())})
        return output
    
    elif (request.method == 'GET'):
        
        return jsonify({'Status':'Running...'})

    else:
        
        print('Invalid method.')
        
        return '{}'
    
if __name__ == '__main__':
    app.run(debug=True)