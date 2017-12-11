'''
    word_embeddings.py
    @author Ruijiang Gao
    
    Get word embeddings from data.
'''



import re
import nltk
import numpy as np
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
	return model, w2v


class sixfourEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(word2vec.values())[0])
    def fit(self, X, y):
        return self
    def transform(self, X):
        vec = np.array([np.array([self.word2vec[words[w]] if w<(len(words))and words[w] in self.word2vec else np.array(np.zeros(self.dim)) for w in range(32)]) for words in X])
        return vec


def get(train, test):
    train['news'] = train['news'].map(
            lambda x: [nltk.word_tokenize(i)[0]
                    for i in re.findall("\S+",x) if re.search('\w',i)!=None])
    test['news'] = test['news'].map(
            lambda x: [nltk.word_tokenize(i)[0]
                    for i in re.findall("\S+",x) if re.search('\w',i)!=None])
    train_model, w2v = word2vec(train, 'train')
    Emb = sixfourEmbeddingVectorizer(w2v)
    train_embedding = Emb.transform(train.news)
    test_embedding = Emb.transform(test.news)
    return train_embedding, test_embedding
