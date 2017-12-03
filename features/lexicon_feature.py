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



def load_lex(path = '/media/ruijiang/Windows/E/umich/NLP/project /dataset/LIWC.all.txt'):
	with open(path) as f:
		lex = f.read().splitlines()	
	lex = pd.DataFrame([i.split(' ,') for i in lex])
	return lex

def lex_feature(data, lex):
	data = data.copy()
	data['news'] = data['news'].map(lambda x: [i for i in re.findall("\S+",x) if re.search('\w',i)!=None])
	for cl in set(lex[1]):
		words = lex[lex[1]==cl][0]
		data[cl] = data.news.map(lambda x: sum([1. for j in words for k in x if re.match(j,k)]))
	return data

def word2vec(data,filename):
	model = gensim.models.Word2Vec(data.news.tolist(),size=100)
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	model.wv.save_word2vec_format('word_embedding_'+filename+'.txt', binary=False)
	return model, w2v

def unigrams(data, test, valid):
	data = data.copy()
	Vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df = 20)
	unig = Vectorizer.fit_transform(data.news)
	train_f = pd.DataFrame(unig.todense(),columns = Vectorizer.get_feature_names())
	test_f = pd.DataFrame(Vectorizer.transform(test.news).todense(),columns = Vectorizer.get_feature_names())
	valid_f = pd.DataFrame(Vectorizer.transform(valid.news).todense(),columns = Vectorizer.get_feature_names())
	return train_f, test_f, valid_f

def load(path):
	data = pd.read_csv(path,header = None)
	data = data[[1,2]]
	data.columns = ['labels','news']
	data['news'] = data['news'].map(lambda x: x.lower().translate(dict((ord(char), None) for char in string.punctuation if char!='\'' and char!='-')) )  
	data['news'] = data['news'].map(lambda x: x.replace('\n',' '))
	return data

LIWC = load_lex()

train = load('train.csv')
test = load('test.csv')
valid = load('valid.csv')

######    unigram 
train_uni, test_uni, valid_uni = unigrams(train, test, valid)
###### lexicon: LIWC
LIWC_f = lex_feature(data, LIWC)