import sys

from choose_features import load_features
from choose_classifier import run_classifier
from choose_data import load_data
from choose_output import write_out_file


'''
    public functions
'''
def get_command_arguments():
    return sys.argv[1], sorted(sys.argv[2:])


def pipeline(classifier, features):
    train_data, test_data = load_data()
    train_features, test_features = load_features(train_data, test_data)
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
