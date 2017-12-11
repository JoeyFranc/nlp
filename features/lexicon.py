'''
    lexicon.py
    @author Ruijiang Gao
    
    Features from the LIWC lexicon.
'''



import re
import pandas as pd



PATH_DICTIONARY = {
    'liwc': '/media/ruijiang/Windows/E/umich/NLP/project /dataset/LIWC.all.txt',
    'moral':,
    'affect':
}

def load_lex(path = get_liwc_path()):
	with open(path) as f:
		lex = f.read().splitlines()
	lex = pd.DataFrame([i.split(' ,') for i in lex])
	return lex


def lex_feature(data,lex):
    data = data.copy()
    data['news'] = data['news'].map(lambda x: [i.lower() for i in re.findall("\S+",x) if re.search('\w',i)!=None])
    for cl in set(lex[1]):
        words = lex[lex[1]==cl][0]
        data[cl] = data.news.map(lambda x: sum([1. for j in words for k in x if re.match('^'+j.replace('*','.+')+'$',k)]))
    return data


def get(train, test)
    LIWC = load_lex()
    return lex_feature(train, LIWC), lex_feature(test, LIWC)
