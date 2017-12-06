import numpy as np
import random
import keras
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge, Conv2D, Convolution1D, MaxPooling2D, ZeroPadding2D, LSTM,Bidirectional, Concatenate
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras import backend as KK
#X_train1 = np.random.random((200, 100, 80,1))
#X_valid1 = np.random.random((100, 100, 80,1))
#X_train2 = np.random.random((200, 5,1))
#X_valid2 = np.random.random((100, 5,1))
#label1 = np.random.randint(6, size=(200, 1))
#label2 = np.random.randint(6, size=(100, 1))

# Convert labels to categorical one-hot encoding

#Y_valid = keras.utils.to_categorical(label2, num_classes=6)


def create_model(F, K, C, T):
    first = Sequential()
    first.add(Conv2D(filters = 32, kernel_size = (3, 3),input_shape=(F,K,1), activation = 'relu'))
    first.add(MaxPooling2D((2, 2)))
    first.add(Flatten())
    second = Sequential()
    second.add(Convolution1D(16, kernel_size = 2, input_shape = (C,1), activation='relu'))
    second.add(Bidirectional(LSTM(8)))
    model = Sequential()
    model.add(Merge([first, second], mode='concat'))
    model.add(Dense(T, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(T, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="sgd",
                  metrics=['accuracy'])
    return model

#suppose word_embedding _matrix n * f * k, feature_matrix n* c, label with one-hot encoding n*t
#Y_train = keras.utils.to_categorical(label1, num_classes=6) for one-hot encoding

def hcnn(word_embedding_matrix, feature_matrix, label_matrix):
    batch_size = 4
    nb_epoch = 5
    N, F, K = word_embedding_matrix.shape
    C = feature_matrix.shape[1]
    T = label_matrix.shape[1]
    X1 = word_embedding_matrix.reshape(N, F, K, 1)
    X2 = feature_matrix.reshape(N, C, 1)
    random.seed(123)
    random_ind = random.sample(range(N), N)
    train_ind = random_ind[0:int(N*0.8)]
    valid_ind = random_ind[int(N*0.8):N]
    model = create_model(F, K, C, T)
    callbacks = [EarlyStopping(monitor='val_loss', patience=nb_epoch, verbose=0)]
    history = model.fit([X1[train_ind,:,:,:], X2[train_ind,:,:]], label_matrix[train_ind,:], batch_size=batch_size, nb_epoch=nb_epoch, \
                        shuffle=True, verbose=2, validation_data=([X1[valid_ind,:,:,:], X2[valid_ind,:,:]], label_matrix[valid_ind,:]), \
                        callbacks=callbacks)  # record history by JH
    return (model.predict([X1, X2], batch_size=batch_size, verbose=2))
	

