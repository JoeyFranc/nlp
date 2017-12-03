import numpy as np
np.random.seed(801)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
import keras
warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version



def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()
    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('/media/ruijiang/Windows/E/umich/stat503/project/keras/train', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    print('Loading test data')
    path = os.path.join('/media/ruijiang/Windows/E/umich/stat503/project/keras/train', 'test', '*.jpg')
    files = sorted(glob.glob(path))
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    X_test = []
    X_test_id = []
    Y_test=[]
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('/media/ruijiang/Windows/E/umich/stat503/project/keras/train', 'test', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_test.append(img)
            X_test_id.append(flbase)
            Y_test.append(index)
    return X_test, X_test_id, Y_test
    

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


#xuyao chuli zheli  pca normalization  augmentation?
def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))
    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id, test_target = load_test()
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))
    test_data = test_data.astype('float32')
    test_data = test_data / 255
    test_target = np_utils.to_categorical(test_target, 8)
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id, test_target


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


#xuyao chuli zheli 
def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


#xuyao chuli zheli relu dropout softmax sgd
def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64), dim_ordering='th', name = 'pad1'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', name = 'conv1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th', name = 'pad2'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', name = 'conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th', name = 'maxpool1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th', name = 'pad3'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', name = 'conv3'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th', name = 'pad4'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', name = 'conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th', name = 'maxpool2'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    return model


#xuyaochulizheli
def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def output_history(historys, nb_epoch):
    nfolds = len(historys)
    acc = np.linspace(0,0,num=nb_epoch)
    loss = np.linspace(0,0,num=nb_epoch)
    val_acc = np.linspace(0,0,num=nb_epoch)
    val_loss = np.linspace(0,0,num=nb_epoch)
    for i in range(nfolds):
            acc += np.array(historys[i].history['acc'])
            loss += np.array(historys[i].history['loss'])
            val_acc += np.array(historys[i].history['val_acc'])
            val_loss += np.array(historys[i].history['val_loss'])
    history_df = pd.DataFrame({'epoch':np.array(range(nb_epoch))+1, 'acc':acc/nfolds, 'loss':loss/nfolds, 'val_acc':val_acc/nfolds, 'val_loss':val_loss/nfolds})
    history_df.to_csv("train_df.csv", sep = ',', index = False)



#xuyao chuli zheli
def run_cross_validation_create_models(nfolds=5):
    # input image dimensions
    batch_size = 20
    nb_epoch = 100
    random_state = 51
    train_data, train_target, train_id = read_and_normalize_train_data() 
    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)  #default 5 fold
    num_fold = 0
    sum_score = 0
    models = []
    historys = []  #append history item by JH
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=nb_epoch, verbose=0),
        ]
        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)  #record history by JH
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)
        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]
        models.append(model)
        historys.append(history)
    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)
    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models, historys, nb_epoch

#xuyao chuli zheli
def run_cross_validation_process_test(info_string, models):
    batch_size = 20
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)
    test_data, test_id, test_target= read_and_normalize_test_data()
    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)
    test_res = merge_several_folds_mean(yfull_test, nfolds)  #predict test based on all folds of model by JH
    test_y=[]
    for i in range(len(test_res)):
        temp=max(enumerate(test_res[i]),key=lambda x: x[1])[0]
        test_y.append(temp)
    target=[]
    for i in range(len(test_target)):
        temp=max(enumerate(test_target[i]),key=lambda x: x[1])[0]
        target.append(temp)
    print("test y:")
    print(test_y)
    print("target:")
    print(target)
    print("Final Accuracy:")
    print(float(sum([1 for j in range(len(test_y)) if test_y[j]==target[j]]))/len(test_y))
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)

def show_image(num, border):
 folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
 nimgs = num
 imgs = np.zeros((num, 128, 128, 3), dtype=np.int)
 imshape = imgs.shape[1:]
 paddedh = imshape[0] + border
 paddedw = imshape[1] + border
 nrows = len(folders)
 ncols = nimgs
 bigimg = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border, 3),
                            dtype=np.int)
 row=0
 for type in folders:
  path = os.path.join('/media/ruijiang/Windows/E/umich/stat503/project/keras/train', 'train', type, '*.jpg')
  files = glob.glob(path)[:(nimgs+1)]
  for i in range(nimgs):
   col = i
   img = cv2.imread(files[i])
   resize = cv2.resize(img, (64,64), cv2.INTER_LINEAR)
   resize=[resize]
   resize = np.array(resize, dtype=np.uint8)
   resize = resize.transpose((0, 3, 1, 2))
   resize = resize.astype('float32')
   conv_fish=models[0].predict(resize)

   bigimg[row * paddedh:row * paddedh + imshape[0],col * paddedw:col * paddedw + imshape[1],0:3] = resize
  row=row+1
 return bigimg

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models, historys, nb_epoch = run_cross_validation_create_models(num_folds)
    output_history(historys, nb_epoch)
    run_cross_validation_process_test(info_string, models)