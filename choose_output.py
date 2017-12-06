import numpy as np

from utils import get_out_file_name



def calculate_acccuracy(result_matrix, true_labels):
    labels = [np.argmax(p) for p in result_matrix]
    correct = np.sum(np.equal(result_matrix, true_labels))
    return correct / len(labels)


def get_line(prob_vector):
    output = ''
    for number in prob_vector:
        output += str(number) + ', '
    output += str(np.argmax(prob_vector))
    return output


def write_out_file(result_matrix, classifier, features):
    file_name = get_out_file_name(classifier, features)
    with open(file_name, 'w') as out_file:
        out_file.write(str(calculate_accuracy) + '\n')
        for prob_vector in result_matrix:
            out_file.write(get_line(prob_vector) + '\n')
