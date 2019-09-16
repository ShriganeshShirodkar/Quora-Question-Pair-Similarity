from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
#from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
#%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc
from scipy import sparse
import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
#%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
# This package is used for finding longest common subsequence between two strings
# you can write your own dp code for this
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
# Import the Required lib packages for WORD-Cloud generation
# https://stackoverflow.com/questions/45625434/how-to-install-wordcloud-in-python3-6
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image
import spacy



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
#df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
#df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
def feat_1(dfx):
    
    dfx['q1len'] = dfx['question1'].str.len() 
    dfx['q2len'] = dfx['question2'].str.len()
    dfx['q1_n_words'] = dfx['question1'].apply(lambda row: len(row.split(" ")))
    dfx['q2_n_words'] = dfx['question2'].apply(lambda row: len(row.split(" ")))

    def normalized_word_Common(row):

        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)
    dfx['word_Common'] = dfx.apply(normalized_word_Common, axis=1)

    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * (len(w1) + len(w2))
    dfx['word_Total'] = dfx.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    dfx['word_share'] = dfx.apply(normalized_word_share, axis=1)

    #dfx['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    #dfx['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])


    
    return dfx

###################################################

# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
    
#===================================================================
def get_token_features(q1, q2):
    token_features = [0.0]*10
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(dfx):
    # preprocessing each question
    dfx["question1"] = dfx["question1"].fillna("").apply(preprocess)
    dfx["question2"] = dfx["question2"].fillna("").apply(preprocess)

    print("token features...")
    
    # Merging Features with dataset
    
    token_features = dfx.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
    dfx["cwc_min"]       = list(map(lambda x: x[0], token_features))
    dfx["cwc_max"]       = list(map(lambda x: x[1], token_features))
    dfx["csc_min"]       = list(map(lambda x: x[2], token_features))
    dfx["csc_max"]       = list(map(lambda x: x[3], token_features))
    dfx["ctc_min"]       = list(map(lambda x: x[4], token_features))
    dfx["ctc_max"]       = list(map(lambda x: x[5], token_features))
    dfx["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    dfx["first_word_eq"] = list(map(lambda x: x[7], token_features))
    dfx["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    dfx["mean_len"]      = list(map(lambda x: x[9], token_features))
   
    #Computing Fuzzy Features and Merging with Dataset
    
    # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
    # https://github.com/seatgeek/fuzzywuzzy
    print("fuzzy features..")

    dfx["token_set_ratio"]       = dfx.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    dfx["token_sort_ratio"]      = dfx.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    dfx["fuzz_ratio"]            = dfx.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    dfx["fuzz_partial_ratio"]    = dfx.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    dfx["longest_substr_ratio"]  = dfx.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return dfx

#======================================================
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sig_clf = joblib.load('sig_clf.pkl')
    tf_idf_vect = joblib.load('tf_idf_vect.pkl')
    to_predict_list = request.form.to_dict()
    print(to_predict_list)
    question1 = to_predict_list['question1']
    question2 = to_predict_list['question2']
    print(question1,question2)
    dfx=pd.DataFrame([question1],columns=['question1'])
    dfx['question2']=question2
    dfx=feat_1(dfx)
    dfz = extract_features(dfx)
    test=tf_idf_vect.transform(dfz['question1']+dfz['question2'])
    df3=dfz[['cwc_min','cwc_max','csc_min','csc_max','ctc_min','ctc_max','last_word_eq','first_word_eq','abs_len_diff','mean_len','token_set_ratio','token_sort_ratio','fuzz_ratio','fuzz_partial_ratio','longest_substr_ratio','q1len','q2len','q1_n_words','q2_n_words','word_Common','word_Total','word_share']]
    #pred = clf.predict(count_vect.transform([review_text]))
    test=sparse.hstack([test,df3])
    pred=sig_clf.predict_proba(test)
    if pred[0][1]>=0.5:
        prediction = "Questions are similar with probablity of {}".format(pred[0][1]) 
    else:
        prediction = "Questions are not similar with probability  {}".format(pred[0][0]) 

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    
    app.run(debug=True)

    
