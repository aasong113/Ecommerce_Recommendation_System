# Author: Anthony Song
# Python 3.7 Interpreter.
# API for connecting the word2vec product recommendation system algorithm using Flask.
# input the product ID, returns 6 similar products. 

import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import flask
import pickle
from flask import Flask, render_template, request
from gensim.models import Word2Vec
import re

app=Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():


    # load model.
    loaded_model = pickle.load(open('model_api.pkl', 'rb'))

    # load dictionary
    loaded_dictionary = pickle.load(open('products_dict.pkl', 'rb'))

    # we used the POST method to transport the form data to the server in the message body.
    # This assumes correct input from a value that is in the dictionary. All you need to input is the product tag.
    if request.method == 'POST':

        # This is the product ID of interest.
        message = request.form['message']

        #remove all uneccesary spaces and non-alphanumeric characters.
        product_id = re.sub('[^0-9a-zA-Z]+', '', message)

        # gets the embedding from the loaded word2vec model.
        vector = loaded_model.wv[product_id]

        # extract most similar products for the input vector
        n = 6
        ms = loaded_model.wv.similar_by_vector(vector, topn=n + 1)

        # extract name and similarity score of the similar products
        new_ms = []
        for j in ms:
            pair = (loaded_dictionary[j[0]][0], j[1])
            new_ms.append(pair)

        return render_template('predict.html', prediction=new_ms)

if __name__ == '__main__':
    app.run(debug=True)
