"""
    choose_features.py
    @author Joey Franc
    
    Using input arguments, returns the input for the classifier.
"""
    
import os
import numpy as np

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
    if original:
        try:
            return np.append(original, tensor, axis=axis)
        except ValueError:
            raise InconsistentTensorError(original, new_tensor)
    else:
        return new_tensor


'''
    public
'''
def get_features(feature):
    """
    About:
        Learns features for the first time by running their accompanying
        "feature.py" file.
    Input:
        feature (str) - The name of the feature to calculate and store.
    Ouput:
        (np.array) - The feature matrix for this feature.
    """
    feature_module = import_feature(feature)
    return feature_module.get()

    
def write_feature_file(feature):
    """
    About:
        Dumps a pickle (overwriting if necessary) representing the
        output for this feature.
    Input:
        feature (str) - The name of the feature to calculate and store.
    Ouput:
        (np.array) - The feature matrix for this feature.
    """
    output = get_features(feature)
    with open(get_feature_file_name(feature), 'wb') as out_file:
        pickle.dump(output, out_file)
    return output
            
            
def load_feature(feature):
    """
    About:
        Gives an np array representing the features for all n
        data points.  Only learns features if necessary.
    Input:
        feature (str) - A valid feature name to be loaded.
    Output:    
        returns (np.array) - An np array of unknown dimension.
    """
    # Infer file name with stored data
    file_name = get_feature_file_name(feature)
    # Write the file if it doesn't exist
    if not os.path.exists(file_name):
        return write_feature_file(feature)
    # Return the np array
    with open(file_name, 'r') as feature_file:
        return pickle.load(feature_file)


def load_features(features):
    """
    About:
        Loads all features and organizes them into two groups:
        1. Word Embeddings
        2. Statement Features
    Input:
        features (list<str>) - A list of valid feature names to load.
    """
    word_embeddings = []
    statement_features = []
    for feature in features:
        tensor = load_feature(feature)
        # This is a word_embedding
        if len(tensor.shape) is 3:
            word_embeddings = _tensor_append(
                    word_embeddings,
                    tensor,
                    feature,
                    axis=2)
        # This is a simple feature
        elif len(tensor.shape) is 2:
            statement_features = _tensor_append(
                    statement_features,
                    tensor,
                    feature,
                    axis=1)
        # This is an error
        else:
            raise InvalidTensorError(tensor)
    return word_embeddings, statement_features
