"""
    choose_features.py
    @author Joey Franc
    
    Using input arguments, returns the input for the classifier.
"""
    


import numpy as np
import os
import pickle

from exceptions import InconsistentTensorError
from exceptions import InvalidTensorError
from exceptions import MissingFeatureError
from utils import import_feature
from utils import is_valid_classifier
from utils import is_valid_feature
from utils import get_feature_file_name



'''
    private
'''
def _validate_input(classifier, features):
    if not is_valid_classifier(classifier):
        raise MissingClassifierError(classifier)
    for feature in features:
        if not is_valid_feature(feature):
            raise MissingFeatureError(feature)


def _tensor_append(original, new_tensor, feature, axis):
    if original[0]:
        try:
            return (np.append(original[0], new_tensor[0], axis=axis),
                    np.append(original[1], new_tensor[1], axis=axis))
        except ValueError:
            raise InconsistentTensorError(original, new_tensor, feature)
    else:
        return new_tensor


'''
    public
'''
def get_features(feature, train_data, test_data):
    """
    About:
        Learns features for the first time by running their accompanying
        "feature.py" file.
    Input:
        feature (str) - The name of the feature to calculate and store.
        train_data (pd) - Data in training set
        test_data (pd) - Data in test set
    Ouput:
        (np.array) - The feature matrix for this feature.
    """
    feature_module = import_feature(feature)
    return feature_module.get(train_data, test_data)

    
def write_feature_file(feature, train_data, test_data):
    """
    About:
        Dumps a pickle (overwriting if necessary) representing the
        output for this feature.
    Input:
        feature (str) - The name of the feature to calculate and store.
        train_data (pd) - Data in training set
        test_data (pd) - Data in test set
    Ouput:
        (np.array) - The feature matrix for this feature.
    """
    output = get_features(feature, train_data, test_data)
    with open(get_feature_file_name(feature), 'wb') as out_file:
        pickle.dump(output, out_file)
    return output
            
            
def load_feature(feature, train_data, test_data):
    """
    About:
        Gives an np array representing the features for all n
        data points.  Only learns features if necessary.
    Input:
        feature (str) - A valid feature name to be loaded.
        train_data (pd) - Data in training set
        test_data (pd) - Data in test set
    Output:    
        returns (np.array) - An np array of unknown dimension.
    """
    # Infer file name with stored data
    file_name = get_feature_file_name(feature)
    # Write the file if it doesn't exist
    if not os.path.exists(file_name):
        return write_feature_file(feature, train_data, test_data)
    # Return the np array
    with open(file_name, 'rb') as feature_file:
        return pickle.load(feature_file)


def load_features(features, train_data, test_data):
    """
    About:
        Loads all features and organizes them into two groups:
        1. Word Embeddings
        2. Statement Features
    Input:
        features (list<str>) - A list of valid feature names to load.
        train_data (pd) - Data in training set
        test_data (pd) - Data in test set
    Returns:
        A training and test (tuple) of the form:
        (word_embeddings, statement_features)
    """
    word_embeddings = ([],[])
    statement_features = ([],[])
    for feature in features:
        tensor = load_feature(feature, train_data, test_data)
        # This is a word_embedding
        if len(tensor[0].shape) is 3:
            word_embeddings = _tensor_append(
                    word_embeddings,
                    tensor,
                    feature,
                    axis=2)
        # This is a simple feature
        elif len(tensor[0].shape) is 2:
            statement_features = _tensor_append(
                    statement_features,
                    tensor,
                    feature,
                    axis=1)
        # This is an error
        else:
            raise InvalidTensorError(tensor)
    return word_embeddings, statement_features
