#import statments
import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
import math
import numpy as np
import multiprocessing as mp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, metrics, svm
from sklearn.utils import shuffle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import pickle

from datetime import datetime
#from tensorflow.python.tools import freeze_graph
import tensorflow as tf

pd.options.mode.chained_assignment = None

import nltk
nltk.download('stopwords')
nltk.download('brown')
nltk.download('names')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords

#from normalise import normalise
##import en_core_web_sm
#nlp = en_core_web_sm.load()

class CNN(object):
    
    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        
        '''
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess =  tf.compat.v1.InteractiveSession(graph = self.graph)

        with tf.io.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read(n=-1))

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        # Define input tensor
        self.input = tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
        self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')

        tf.import_graph_def(graph_def, {'input': self.input, 'dropout_rate': self.dropout_rate})

        print('Model loading complete!')

        '''
        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        '''

        '''
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        '''

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/cnn/output:0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: data, self.dropout_rate: 0})

        return output


class GloveVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        #return np.array([self.nlp(text).vector for text in X])
        return np.array([text.vector for text in X])
class TextProcessor():
  def __init__(self, text_df):
    self.lemmatizer = WordNetLemmatizer()
    cnt = Counter()
    for text in text_df["Description"].values:
      for word in text.split():
        if word.lower() not in stopwords.words('english'):
          cnt[word] += 1
    n_words = 10      
    self.most_frequent_words = set([w for (w, wc) in cnt.most_common(n_words)])
    print(f'Top {n_words} frequent words : {self.most_frequent_words}')
    self.most_infrequent_words = set([w for (w, wc) in cnt.most_common()[:-n_words-1:-1]])
    print(f'Top {n_words} rare words : {self.most_infrequent_words}')    


  def remove_punctuation(self, text):    
    return text.translate(str.maketrans(' ', ' ', string.punctuation))

  def remove_names(self, text):    
    # print(text)
    orig_words_list = text.split()
    tagged_sentence = nltk.tag.pos_tag(orig_words_list)
    word_list = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    print(f'Removed proper noun(s) : {set(orig_words_list)-set(word_list)}')
    return ' '.join(word for word in word_list)

  def remove_words(self, text, removable_words):
    return " ".join([word for word in text.split() if word not in removable_words])    

  def preprocess(self, text_df):
    print("Removing proper nouns") 
    # remove names - like Anthony, cristóbal, eduardo eric fernández
    # TODO - check whether this is removing too many words, especially the ones starting with capital letter
    text_df["Description"] = text_df["Description"].apply(lambda text: self.remove_names(text))

    print("Converting to lower case")
    text_df["Description"] = text_df["Description"].str.lower()

    print("Removing standard punctuations")
    text_df["Description"] = text_df["Description"].apply(lambda text: self.remove_punctuation(text))

    print("Removing Stopwords")
    #EXCLUDED_REMOVE_WORDS={'hand'}
    rem_words_set = {"a", "an", "cm", "kg", "mr", "wa" ,"nv", "ore", "da", "pm", "am", "cx"}
    # remove most frequent and most infrequent words, to experiment
    # words_to_remove = rem_words_set.union(set(stopwords.words('english'))).union(self.most_frequent_words).union(self.most_infrequent_words).difference(EXCLUDED_REMOVE_WORDS)
    words_to_remove = rem_words_set.union(set(stopwords.words('english')))
    print(f"Removing {words_to_remove}")
    text_df["Description"] = text_df["Description"].apply(lambda text: self.remove_words(text, words_to_remove))

    print("Lemmatizing")
    #text_df["Description"] = text_df["Description"].apply(lambda text: ' '.join([t.lemma_ for t in text]))

    print("Removing words containing numbers - like cx695, 945")
    text_df["Description"] = text_df["Description"].apply(lambda text: ' '.join(s for s in text.split() if not any(c.isdigit() for c in s)))
    
    return text_df
## DATA FRAME PRE PROCESSING FUNCTION
# extracts month and day of week from date
#def extract_date(df):  
  #df['month'] = pd.DatetimeIndex(df['Data']).strftime('%B')
  #df['day_of_week'] = pd.DatetimeIndex(df['Data']).strftime('%A')  
 # return df

# Replaces space to make a single word in select columns
def replace_space(df, cols=['Employee or Third Party', 'Critical Risk']):
  for col in cols:
    df[col] = df[col].str.replace('\\(', '')
    df[col] = df[col].str.replace('\\)', '')
    df[col] = df[col].replace(' ', '_', regex=False)
  return df

# Concatenates other columns to description
def concat_cols_to_text(df):
  concat_cols = ['Countries', 'Local', 'Industry Sector', 'Genre', 'Employee or Third Party', 'Critical Risk', 'month', 'day_of_week', 'Description']
  df['text'] = df[concat_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
  return df  

# Exports the df to new csv file 
def export_to_csv(new_df, filepath='/content/sample_data/capstone_input_text_preprocessed.csv'):
  cols=['text', 'Potential Accident Level']
  print(f'Exporting columns {cols} to new csv file : {filepath}')
  new_df[cols].to_csv(filepath)
