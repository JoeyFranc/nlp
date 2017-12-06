'''
    lbc.py
    @author Joey Franc
    
    Linguistic-Based Cues (LBC) used as statement feature representation.
    18 different features in total.
'''

import numpy as np
import nltk

import pandas as pd
import string
import re
import nltk
import scipy
import sklearn.feature_extraction.text
import numpy as np

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import linear_model
from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.util import ngrams
from sklearn.preprocessing import OneHotEncoder

def load(path):
    data = pd.read_csv(path,header = None)
    data = data[[1,2]]
    data.columns = ['labels','news']
    data['news'] = data['news'].map(lambda x: x.lower().translate(dict((ord(char), None) for char in string.punctuation if char!='\'' and char!='-')) )  
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
    data['news'] = data['news'].map(lambda x: [i.lower() for i in re.findall("\S+",x) if re.search('\w',i)!=None])
    for cl in set(lex[1]):
        words = lex[lex[1]==cl][0]
        data[cl] = data.news.map(lambda x: sum([1. for j in words for k in x if re.match('^'+j.replace('*','.+')+'$',k)]))
    return data

def unigrams(data, test):
    data = data.copy()
    Vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df = 20)
    unig = Vectorizer.fit_transform(data.news)
    train_f = pd.DataFrame(unig.todense(),columns = Vectorizer.get_feature_names())
    test_f = pd.DataFrame(Vectorizer.transform(test.news).todense(),columns = Vectorizer.get_feature_names())
    return train_f, test_f



THIRD_PERSON_PRONOUNS = [
    'he', 'she', 'it',
    "he'd", "she'd",
    "he'll", "she'll", "it'll",
    "he's", "she's", "it's",
    'his', 'hers', 'him', 'its',
    'they', "they're", 'them', 'their']

FIRST_PERSON_SINGULAR_PRONOUNS = ['i', "i'm", "i'd", "i'll"]
FIRST_PERSON_PLURAL_PRONOUNS = ['we', "we've", "we'll"]
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
ADJECTIVES = ['JJ', 'JJR', 'JJS']
ADVERBS = ['RB', 'RBR', 'RBS']
FUNCTION_WORDS = ['CC', 'DT', 'PDT', 'RP', 'WDT']


def _get_num_occurances(set_1, set_2):
    return sum([1 for e in set_2 if e in set_1])


class DataPoint(object):
    
    def __init__(self, datapoint):
        self.dp = datapoint
        self.words = nltk.word_tokenize(self.dp)
        self.tagged = nltk.pos_tag(self.words)
        self.pos = [tag for word, tag in self.tagged]
        self.tree = nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}').parse(self.tagged)
        
    
    def _get_num_verbs(self):
        return _get_num_occurances(VERBS, self.pos)
    
    
    def _get_num_nouns(self):
        return _get_num_occurances(NOUNS, self.pos)
    
    
#    def _get_num_clauses(self):
#        self.tree = nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}').parse(self.tagged)
#        return 
    
    
    def _get_num_sentences(self):
        return len(nltk.sent_tokenize(self.dp))
    
    
    def _get_num_words(self):
        return len(self.words)
    
    
    def _get_num_char(self):
        return len(self.dp)
    
    
    def _get_num_words_in_noun_phrases(self):
        count = 0
        for subtree in self.tree.subtrees(lambda t: t.label() == 'NP'):
            count += len(subtree.leaves())
        return count
    
    
    def _get_num_noun_phrases(self):
        return len(list(self.tree.subtrees(lambda t: t.label() == 'NP')))
    
    
    def _get_num_punctuation_marks(self):
        return _get_num_occurances(string.punctuation, self.dp)


    def _get_num_modifiers(self):
        return _get_num_occurances(ADVERBS+ADJECTIVES, self.pos)
    
    
    def _get_num_modal_verbs(self):
        return _get_num_occurances(['MD'], self.pos)
    
    
    def _get_num_uncertainty(self):
        pass
    
    
    def _get_num_other_reference(self):
        return _get_num_occurances(THIRD_PERSON_PRONOUNS, self.words)
    
    
    def _get_self_reference(self):
        return _get_num_occurances(FIRST_PERSON_SINGULAR_PRONOUNS, self.words)
    
    
    def _get_group_reference(self):
        return _get_num_occurances(FIRST_PERSON_PLURAL_PRONOUNS, self.words)
    
    
    def _get_num_adjectives(self):
        return _get_num_occurances(ADJECTIVES, self.pos)
    
    
    def _get_num_adverbs(self):
        return _get_num_occurances(ADVERBS, self.pos)
    
    
    def _get_num_unique_words(self):
        return len({word for word in self.words})
    
    
    def _get_num_function_words(self):
        return _get_num_occurances(FUNCTION_WORDS, self.pos)
    
    
#    def _get_avg_num_clauses(self):
#        return self._get_num_clauses() / self._get_num_sentences()
    
    
    def _get_avg_sentence_length(self):
        return self._get_num_words() / self._get_num_sentences()
    
    
    def _get_avg_word_length(self):
        return self._get_num_char() / self._get_num_words()
    
    
    def _get_avg_noun_phrase_length(self):
        if self._get_num_noun_phrases() == 0: return 0
        return self._get_num_words_in_noun_phrases() / self._get_num_noun_phrases()
    
    
    def _get_pausality(self):
        return self._get_num_punctuation_marks() / self._get_num_sentences()
    
    
    def _get_emotiveness(self):
        return (self._get_num_adjectives() + self._get_num_adverbs()) / (self._get_num_nouns() + self._get_num_verbs())
    
    
    def _get_lexical_diversity(self):
        return self._get_num_unique_words() / self._get_num_words()
    
    
    def _get_redundancy(self):
        return self._get_num_function_words() / self._get_num_sentences()
    
    
    def get_feature_vector(self):
        return np.array([
                self._get_num_words(),
                self._get_num_verbs(),
                self._get_num_noun_phrases(),
                self._get_num_sentences(),
#               self._get_avg_num_clauses(),
                self._get_avg_sentence_length(),
                self._get_avg_word_length(),
                self._get_avg_noun_phrase_length(),
                self._get_pausality(),
                self._get_num_modifiers(),
                self._get_num_modal_verbs(),
#               self._get_num_uncertainty(),
                self._get_num_other_reference(),
                self._get_self_reference(),
                self._get_group_reference(),
                self._get_emotiveness(),
                self._get_lexical_diversity(),
                self._get_redundancy()
        ])
    

def get_datapoints(data_set):
    return np.vstack([DataPoint(dp).get_feature_vector() for dp in data_set])



def get():
    train_data = load('train.csv')
    test_data = load('test.csv')
    #dev_data = load('dev.csv')
    test_set = get_datapoints(test_data['news'])
    #dev_set = get_datapoints(dev_data)
    train_set = get_datapoints(train_data['news'])
    return train_set, test_set


if __name__ == '__main__':
    get()