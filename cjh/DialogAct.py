#Jiahua Chen uniquename: jiahua
import re
import pandas as pd
import numpy as np
import sys
from collections import Counter
import random



class naive_bayes():
    def __init__(self):
        self.label_list = []
        self.content_list = []
        #self.origin_content_advisor = []
        #self.origin_content_student = []
        self.label_count = Counter()
        self.label_word = {}
        self.train_l = None
    def read_data(self, file_name):
        with open(file_name) as myfile:
            content = myfile.readlines()
        myfile.closed
        alter_content = [re.sub("(\(|\)|\!|\,|\?|\.)", "", item.lower()) for item in \
                             content]
        tmp_content = "SOS"
        #tmp_origin = ""
        for i in range(len(alter_content)):
            tmp = re.findall('(?<=\[).*?(?=\])', alter_content[i])
            if len(tmp) > 0:
                #self.origin_content_advisor.append(content[i])
                #self.origin_content_student.append(tmp_origin)
                self.label_list.append(tmp[0])
                self.content_list.append(tmp_content)
                tmp_content = "SOS"
                #tmp_origin = ""
            else:
                #tmp_origin = tmp_origin + content[i]
                tmp_content = tmp_content + ' ' + re.sub('student: ', '', alter_content[i]).rstrip('\n')
    def train(self):
        train_index = range(len(self.label_list))
        train_label_list = [self.label_list[i] for i in train_index]
        train_content_list = [self.content_list[i] for i in train_index]
        train_len = len(train_index)
        self.train_l = train_len
        label_key = np.unique(self.label_list).tolist()
        for key in label_key:
            self.label_word[key] = Counter()
        for i in range(train_len):
            tmp_label = train_label_list[i]
            tmp_content = train_content_list[i].split()
            self.label_count[tmp_label]+=1
            tmp_counter = Counter(tmp_content)
            for key, value in tmp_counter.items():
                tmp_counter[key] = 1
            self.label_word[tmp_label] = self.label_word[tmp_label]+tmp_counter
    def test(self, file_name):
        with open(file_name) as myfile:
            content = myfile.readlines()
        myfile.closed
        origin_content_advisor = []
        origin_content_student = []
        tmp_origin = ""
        train_len = self.train_l
        alter_content = [re.sub("(\!|\,|\?|\.)", "", item.lower()) for item in \
                   content]
        tmp_content = "SOS"
        test_label_list = []
        test_content_list = []
        file = open(file_name + ".out", "a")
        for i in range(len(alter_content)):
            tmp = re.findall('(?<=\[).*?(?=\])', alter_content[i])
            if len(tmp) > 0:
                origin_content_advisor.append(content[i])
                origin_content_student.append(tmp_origin)
                test_label_list.append(tmp[0])
                test_content_list.append(tmp_content)
                tmp_content = "SOS"
                tmp_origin = ""
            else:
                tmp_origin = tmp_origin + content[i]
                tmp_content = tmp_content + ' ' + re.sub('student: ', '', alter_content[i]).rstrip('\n')
        test_len = len(test_label_list)
        label_key = np.unique(self.label_list).tolist()
        #unique_word_count = {label:len(self.label_word[label].keys()) for label in label_key}
        tmp_counter = Counter()
        for i in range(len(label_key)):
            label = label_key[i]
            tmp_counter = tmp_counter + self.label_word[label]
        all_len = len(tmp_counter.keys())
        pred_label = []
        for i in range(test_len):
            tmp_content = test_content_list[i].split()
            score = []
            #file = open(file_name + ".wsd.out", "a")
            for j in range(len(label_key)):
                label = label_key[j]
                tmp_label_count = self.label_count[label]
                term_1 = [np.log2(float(1+self.label_word[label][word])/(tmp_label_count+all_len)) for word in tmp_content]
                score.append(sum(term_1) + np.log2(float(tmp_label_count)/train_len))
            pred_label.append(label_key[np.argmax(np.array(score))])
            file.write(origin_content_student[i]  + "[" + pred_label[i] + "]" + origin_content_advisor[i])
        file.close()
        self.label_count = Counter()
        self.label_word = {}
        accu = float(sum([pred_label[i] == test_label_list[i] for i in range(test_len)]))/test_len
        print('The test accuracy is ' + str(accu) + '\n')


"""
def cross_validation(self, name, fold = 1):
        N = len(self.content_list)
        accuracy = []
        fold = int(fold)
        random.seed(123)
        random_ind = random.sample(range(N), N)
        for k in xrange(fold):
            test_index = [random_ind[i] for i in xrange(N) if i%fold==k]
            train_index = [random_ind[i] for i in xrange(N) if i%fold!=k]
            self.train(train_index)
            file = open(name+".wsd.out", "a")
            file.write("Fold "+str(k+1)+"\n")
            file.close()
            accu = self.test(test_index, name)
            print('The'+str(k+1)+'fold test accuracy is '+str(accu)+'\n')
            accuracy.append(accu)
        print('The average accuracy on test set is ' + str(np.mean(accuracy)))

with open('DialogAct.train') as f:
    content = f.readlines()

label_list = []
content_list = []
tmp_content = "SOS"
for i in range(len(content)):
    tmp = re.findall('(?<=\[).*?(?=\])', content[i])
    if len(tmp)>0:
        label_list.append(tmp[0])
        content_list.append(tmp_content)
        tmp_content = "SOS"
    else:
        tmp_content = tmp_content + ' ' + re.sub('Student: ', '', content[i]).rstrip('\n')

"""



if __name__=="__main__":
    pd.options.mode.chained_assignment = None  # default='warn'
    tmp = naive_bayes()
    tmp.read_data(sys.argv[1])
    tmp.train()
    tmp.test(sys.argv[2])


