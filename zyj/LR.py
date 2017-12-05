import pandas as pd
import string
import re
import nltk
import scipy
import sklearn.feature_extraction.text
import numpy as np

from sklearn import linear_model
from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.util import ngrams

def load(path):
    data = pd.read_csv(path,header = None)
    data = data[[1,2]]
    data.columns = ['labels','news']
    cha = ''
    for char in string.punctuation:
        if char!='\'' and char!='-':
            cha += char
    data['news'] = data['news'].map(lambda x: x.lower().translate(None,cha))
    data['news'] = data['news'].map(lambda x: x.replace('\n',' '))
    return data

# if we want to calculate correlation, we shall rewrite those str lables
def repl(t):
    t.replace('half-true','0.5', inplace=True)
    t.replace('mostly-true','0.8', inplace=True)
    t.replace('barely-true','0.3', inplace=True)
    t.replace('pants-fire','-1', inplace=True)

def lex_feature(data,lex):
    data = data.copy()
    data['news'] = data['news'].map(lambda x: [i for i in re.findall("\S+",x) if re.search('\w',i)!=None])
    for cl in set(lex[1]):
        words = lex[lex[1]==cl][0]
        data[cl] = data.news.map(lambda x: sum([1. for j in words for k in x if re.match(k,j)]))
    return data

def unigrams(data, test):
    data = data.copy()
    Vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df = 20)
    unig = Vectorizer.fit_transform(data.news)
    train_f = pd.DataFrame(unig.todense(),columns = Vectorizer.get_feature_names())
    test_f = pd.DataFrame(Vectorizer.transform(test.news).todense(),columns = Vectorizer.get_feature_names())
    return train_f, test_f

train = load('train.csv')
test = load('test.csv')
# repl(test.iloc[:,0])
# repl(train.iloc[:,0])
liwc = pd.read_csv('LIWC.all.txt',header=None)
affect = pd.read_csv('WordNetAffect.all.txt',header=None)
moral = pd.read_csv('Morality.All.txt',header=None)

# lexicon features [time-consuming, these feature docs have be included in dir /features doc]
# train_liwc_f = lex_feature(train,liwc)
# test_liwc_f = lex_feature(test,liwc)
# train_affect_f = lex_feature(train,affect)
# test_affect_f = lex_feature(test,affect)
# train_moral_f = lex_feature(train,moral)
# test_moral_f = lex_feature(test,moral)
train_liwc_f = pd.read_csv('train_liwc_f',header=None)
test_liwc_f = pd.read_csv('test_liwc_f',header=None)
train_affect_f = pd.read_csv('train_affect_f ',header=None)
test_affect_f = pd.read_csv('test_affect_f ',header=None)
train_moral_f = pd.read_csv('train_moral_f',header=None)
test_moral_f = pd.read_csv('test_moral_f',header=None)

# unigram
train_uni,test_uni = unigrams(train,test)

logreg = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs',C=1e5,max_iter=1000)
# LR on one feature set
# using unigram features
logreg.fit(train_uni,train.iloc[:,0])
pred = logreg.predict(test_uni)
np.mean([pred[i]==test.iloc[i,0] for i in range(len(pred))])
# using lexical features
logreg.fit(train_affect_f.iloc[:,2:],train.iloc[:,0])
pred = logreg.predict(test_affect_f.iloc[:,2:])
np.mean([pred[i]==test.iloc[i,0] for i in range(len(pred))])

# LR based on uni+liwc features
logreg.fit(pd.concat([train_uni, train_liwc_f.iloc[:,2:]], axis=1),train.iloc[:,0])
pred = logreg.predict(pd.concat([test_uni, test_liwc_f.iloc[:,2:]], axis=1))
np.mean([pred[i]==test.iloc[i,0] for i in range(len(pred))])

# LR on all features
f_train = pd.concat([train_uni, train_liwc_f.iloc[:,2:], train_affect_f.iloc[:,2:],train_moral_f.iloc[:,2:]], axis=1)
f_test = pd.concat([test_uni, test_liwc_f.iloc[:,2:], test_affect_f.iloc[:,2:],test_moral_f.iloc[:,2:]], axis=1)

logreg.fit(f_train,train.iloc[:,0])
pred = logreg.predict(f_test)
np.mean([pred[i]==test.iloc[i,0] for i in range(len(pred))])

# correlation calculation
# from scipy.stats.stats import pearsonr
# print pearsonr(pd.to_numeric(pred),pd.to_numeric(test.iloc[:,0]))