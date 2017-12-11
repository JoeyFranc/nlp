'''
    utils.py
    @author Joey Franc
'''



import os



'''
    Constants
'''
FEATURE_FILE_SUFFIX = 'data'
CLASSIFIER_PATH = 'classifiers'
FEATURE_PATH = 'features'
OUT_FILE_SUFFIX = 'txt'
LABEL_MAP = {
        'pants-fire': 0,
        'false': 1,
        'barely-true': 2,
        'half-true': 3,
        'mostly-true': 4,
        'true': 5}



valid_classifiers = os.listdir(CLASSIFIER_PATH)
valid_features = os.listdir(FEATURE_PATH)



'''
    functions
'''
def get_classifiers():
    return valid_classifiers


def get_features():
    return valid_features


def is_valid_classifier(classifier):
    return (classifier + '.py') in get_classifiers()


def is_valid_feature(feature):
    return (feature + '.py') in get_features()


def get_feature_file_name(feature):
    return feature + '.' + FEATURE_FILE_SUFFIX


def get_out_file_name(classifier, features):
    name = classifier
    features = sorted(features)
    for feature in features:
        name += '_' + feature
    name += '.' + OUT_FILE_SUFFIX
    return name


def get_liwc_path():
    return '/media/ruijiang/Windows/E/umich/NLP/project /dataset/LIWC.all.txt'


def import_feature(feature):
    return __import__(FEATURE_PATH + '.' + feature + '.py')


def import_classifier(classifier):
    return __import__(CLASSIFIER_PATH + '.' + classifier + '.py')
