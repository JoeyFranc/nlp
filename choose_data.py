'''
    choose_data.py
    @author Joey Franc, Ruijiang Gao
    
    Get pandas data frames for the test and train sets.
'''



import numpy as np
import pandas as pd
import string

from utils import get_data_file_name
from utils import map_label



def _load(file_name):
    file_name = get_data_file_name(file_name)  # Go down the right path
    data = pd.read_csv(file_name ,header = None)
    data = data[[1,2]]
    data.columns = ['labels','news']
    data['news'] = data['news'].map(lambda x: x.lower().translate(dict((ord(char), None) for char in string.punctuation if char!='\'' and char!='-')) )  
    data['news'] = data['news'].map(lambda x: x.replace('\n',' '))
    return data


def load_data():
    train = _load('train.csv')
    test = _load('test.csv')
    return train, test

def get_labels(train_data, test_data):
    y_train = np.array([map_label(label)
            for label in train_data['labels']])
    y_test = np.array([map_label(label)
            for label in test_data['labels']])
    return (y_train, y_test)
        
