'''
    SVM.py
    @author Ruijiang Gao
    
    A simple Linear SVM classifier.
'''



from sklearn.svm import LinearSVC



def run_classifier(X_train, X_test, y_train, y_test):
    _, statement_features_train = X_train
    _, statement_features_test = X_test
    param_grid = {'C': [1, 1e2, 1e-2, 1e3, 1e4],}
    clf = GridSearchCV(LinearSVC(), param_grid)
    clf.fit(statement_features_train, y_train)
    return clf.score(statement_features_test, y_test)
