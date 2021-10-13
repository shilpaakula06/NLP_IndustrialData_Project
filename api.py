# !/usr/bin/env python
import sys
import logging
from flask import Flask, request, jsonify, render_template  # from random import randrange
from pre_processing import TextProcessor
from tensorflow.keras.preprocessing.text import Tokenizer

# to split Train and Test data from sklearn.model_selection import train_test_split

# To pad sentence #
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import numpy as np
# import pickle 
# import joblib 
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.base import TransformerMixin, BaseEstimator
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import model_selection, metrics, svm
# from sklearn.utils import shuffle
# from keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model  # ,Model
import json
import random
# intents = json.loads(open('D:/GREAT LAKES/CapstoneProject-NLP/UI/static/json/Intent.json').read())
model_pl = load_model('chatbot_model.h5',compile = False)
model_al = load_model('chatbot_model_al.h5',compile = False)

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def init_logger():
    logger = logging.getLogger("model_serve")

    formatter = logging.Formatter('[%(levelname)s] %(message)s')

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

def get_dummy_prediction(x):
  x['text'] = x['Description']  # .apply(lambda x : clean_text(x))
  vocab_size=2243
  maxlen=104
  tokenizer = Tokenizer(num_words=vocab_size)
  text = x['text']
  # Fit the tokenizer object for X_train that contains headlines attrbutes
  tokenizer.fit_on_texts(text)
  # convert text to sequence - sequence encoding for train and test feature - headlines
  train_encoding = tokenizer.texts_to_sequences(text)
  text = pad_sequences(train_encoding, maxlen=maxlen, padding='post') 
  
  predictions = model_pl.predict(text)  
  predictions = np.argmax(predictions, axis = 1)
  Lable={'1','2','3','4','6','7'}
  # print('test2',Lable[predictions])
  print(predictions[0])
  return str(predictions[0])  # np.array(predictions, dtype=np.int64)
# Replaces space to make a single word in select columns
def replace_space(df, cols=['Employee or Third Party', 'Critical Risk']):
  for col in cols:
    df[col] = df[col].str.replace('\\(', '')
    df[col] = df[col].str.replace('\\)', '')
    df[col] = df[col].replace(' ', '_', regex=False)
  return df

# Concatenates other columns to description for potential level prediction
def concat_cols_to_text(df):
  concat_cols = ["Name","Gender","Local","Countries","Industry Sector","Employee or Third Party","Critical Risk", "Description"]
  df['text'] = df[concat_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
  return df  
# Concatenates other columns to description for accident level prediction
def concat_cols_to_text_al(df):
  concat_cols = ["Name","Gender","Local","Countries","Industry Sector","Employee or Third Party","Critical Risk", "Description","Potential Accident Level"]
  df['text'] = df[concat_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
  return df  

def get_accidentlevel_prediction(x):
  x['text'] = x['Description']#.apply(lambda x : clean_text(x))
  vocab_size=20000
  maxlen=100
  tokenizer = Tokenizer(num_words=vocab_size)
  text = x['text']
  # Fit the tokenizer object for X_train that contains headlines attrbutes
  tokenizer.fit_on_texts(text)
  # convert text to sequence - sequence encoding for train and test feature - headlines
  train_encoding = tokenizer.texts_to_sequences(text)
  text = pad_sequences(train_encoding, maxlen=maxlen, padding='post') 
  
  predictions = model_al.predict(text)  
  predictions = np.argmax(predictions, axis = 1)
  Lable={'1','2','3','4','6','7'}
  #print('test2',Lable[predictions])
  print(predictions[0])
  return str(predictions[0])

def get_prediction(features):    
    response = {}
    # TODO
    features=[features]    
    df =  pd.DataFrame.from_dict(features)    
    ppr = TextProcessor(df)
    new_df = ppr.preprocess(df.copy())   
    #upd_df = extract_date(new_df)
    upd_df = replace_space(new_df)
    upd_df_concat = concat_cols_to_text(upd_df) 
    response['risk'] = get_dummy_prediction(upd_df_concat)    
    upd_df.loc[:,'Potential Accident Level'] = response['risk']   
    pot_acc_level = {'1': 'POTACTA', '2': 'POTACTB', '3': 'POTACTC', '4' : 'POTACTD', '5': 'POTACTE'}
    upd_df['Potential Accident Level'] = pot_acc_level[response['risk']]
    upd_df_accidentlevel = concat_cols_to_text_al(upd_df)
    print("get_predictionnew",upd_df.head())
    response['risk_al'] = get_accidentlevel_prediction(upd_df_accidentlevel)
    return response    

logger = init_logger()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():    
    request_json = request.json    
    logger.debug(f'Received request : {request_json}')
    print("classify",request_json)
    response_json = get_prediction(request_json)
    logger.debug(f'Sending response : {response_json}')
    return jsonify(response_json)

if __name__ == '__main__':
    logger.info('Starting model serving api')
    app.run(port=8081)
