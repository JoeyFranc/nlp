from sklearn import svm



def train(word_embeddings, statement_features, labels):
    return svm.SVC().fit(statement_features, labels)


def run(word_embeddings, statement_features, classifier):
    classifier.predict(statement_features)
    
    