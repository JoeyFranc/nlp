import os 
import re
import pandas as pd
import nltk
import string
import os
import numpy as np
import scipy 
import gensim

from collections import Counter
from nltk.stem import PorterStemmer
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.util import ngrams

def word2vec(data,filename):
	model = gensim.models.Word2Vec(data.news.tolist(),size=100)
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	model.wv.save_word2vec_format('word_embedding_'+filename+'.txt', binary=False)
	return model, w2v

def load(path):
	data = pd.read_csv(path,header = None)
	data = data[[1,2]]
	data.columns = ['labels','news']
	data['news'] = data['news'].map(lambda x: x.lower().translate(dict((ord(char), None) for char in string.punctuation if char!='\'' and char!='-')) )  
	data['news'] = data['news'].map(lambda x: x.replace('\n',' '))
	return data

class sixfourEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        vec = np.array([
        	np.array([self.word2vec[words[w]] if w<(len(words))and words[w] in self.word2vec else np.array([np.zeros(self.dim)]) for w in range(64)])
            for words in X
        ])
        return vec

train = load('train.csv')
test = load('test.csv')
valid = load('valid.csv')

train['news'] = train['news'].map(lambda x: [nltk.word_tokenize(i)[0] for i in re.findall("\S+",x) if re.search('\w',i)!=None])
test['news'] = test['news'].map(lambda x: [nltk.word_tokenize(i)[0] for i in re.findall("\S+",x) if re.search('\w',i)!=None])
valid['news'] = valid['news'].map(lambda x: [nltk.word_tokenize(i)[0] for i in re.findall("\S+",x) if re.search('\w',i)!=None])


Emb = sixfourEmbeddingVectorizer(w2v)
train_embedding = Emb.transform(train.news)
test_embedding = Emb.transform(test.news)
valid_embedding = Emb.transform(valid.news)