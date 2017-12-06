"""
    choose_classifier.py
    @author Joey Franc
    
    Runs desired classifier functions with desired inputs.
"""


from utils import import_classifier
from utils import is_valid_classifier
from exceptions import InvalidClassifierError



def _get_classifier(classifier):
    if is_valid_classifier(classifier):
        return __import__(classifier)
    raise InvalidClassifierError(classifier)


def train_classifier(classifier, word_embeddings, statement_features, labels):
    '''
    About:
        Trains a classifier on the statement_features.
    Input:
        classifier (string) - The name of a valid classifier.
        
        word_embeddings (np.array) (n x k x f) - Feature vectors for
        each word in the datapoint.
        
        statement_features (np.array) (n x c) - Feature vectors for
        each datapoint.
    Returns:
        (None) - This saves the newly trained classifier for later use.
    '''
    class_module = _get_classifier(classifier)
    return class_module.train(word_embeddings, statement_features)


def run_classifier(classifier, word_embeddings, statement_features):
    '''
    About:
        Classifies the inputted datapoints.
    Input:
        classifier (string) - The name of a valid classifier.
        
        word_embeddings (np.array) (n x k x f) - Feature vectors for
        each word in the datapoint.
        
        statement_features (np.array) (n x c) - Feature vectors for
        each datapoint.
    Returns:
        (np.array) (n,6) - Probability vector representing the
        likelihood of each datapoint belonging to all 6 classes.
    '''
    if not is_already_trained(classifier, features):
        train_classifier(
                classifier,
                word_embeddings,
                statement_features,
                labels)
    class_module = _get_classifier(classifier)
    return class_module.run(word_embeddings, statement_features)
