'''
    pipeline.py
    @author Joey Franc
    
    This file is the universal .main function, and contains
    techniques for speeding up training and standardizing code.
'''



import sys

from choose_features import load_features
from choose_classifier import run_classifier
from choose_data import load_data
from choose_data import get_labels
from choose_output import write_out_file


'''
    public functions
'''
def get_command_arguments():
    return sys.argv[1], sorted(sys.argv[2:])


def pipeline(classifier, features):
    print('Loading Liar Dataset')
    train_data, test_data = load_data()
    print('Fetching features from dataset')
    word_embeddings, statement_features = load_features(
        features,
        train_data,
        test_data)
    train_embed, test_embed = word_embeddings
    train_state, test_state = statement_features
    train_features = (train_embed, train_state)
    test_features = (test_embed, test_state)
    print('Fetching labels from dataset')
    train_labels, test_labels = get_labels(train_data, test_data)
    print('Running classifier')
    output = run_classifier(
            classifier,
            train_features,
            test_features,
            train_labels,
            test_labels)
#    write_out_file(output, classifier, features)
    

'''
    only invoked when run as a script
'''
if __name__ == '__main__':
    classifier, features = get_command_arguments()
    pipeline(classifier, features)
