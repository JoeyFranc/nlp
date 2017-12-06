import sys

from choose_features import load_features
from choose_classifier import run_classifier
from choose_output import write_out_file
from utils import load_labels



'''
    public functions
'''
def get_command_arguments():
    return sys.argv[1], sys.argv[2] == 'dev', sys.argv[3:]


def pipeline(classifier, features, is_dev):
    test_labels, dev_labels, train_labels = load_labels()
    word_embeddings, statement_features = load_features(features)
    output = run_classifier(
            classifier,
            word_embeddings,
            statement_features,
            labels)
    write_out_file(output, classifier, features)
    

'''
    only invoked when run as a script
'''
if __name__ == '__main__':
    classifier, is_dev, features = get_command_arguments()
    pipeline(classifier, features, is_dev)
