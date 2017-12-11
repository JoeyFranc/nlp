'''
    unigrams.py
    @author Ruijiang Gao
    
    Gets unigram features.
'''



from sklearn.feature_extraction.text import CountVectorizer

def unigrams(data, test):
	data = data.copy()
	Vectorizer = CountVectorizer(min_df = 20)
	unig = Vectorizer.fit_transform(data.news)
	train_f = pd.DataFrame(unig.todense(),columns = Vectorizer.get_feature_names())
	test_f = pd.DataFrame(Vectorizer.transform(test.news).todense(),columns = Vectorizer.get_feature_names())
	return train_f, test_f

def get(train_data, test_data):
    return unigrams(data, test)