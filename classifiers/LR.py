'''
    LR.py
    @author Ruijiang Gao
    
    Simple logistic regression model.
'''


from sklearn import linear_model
from sklearn.model_selection import GridSearchCV



def run_classifier(X_train, X_test, y_train, y_test):
    _, statement_features_train = X_train
    _, statement_features_test = X_test
    lr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)
    param_grid = {'C': [1, 1e2, 1e-2, 1e3,1e4],}
    logreg = GridSearchCV(lr, param_grid)
    logreg.fit(statement_features_train, y_train)
    return logreg.score(statement_features_test, y_test)
