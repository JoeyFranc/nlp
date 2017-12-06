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
from keras import initializers
from keras.layers.normalization import BatchNormalization
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
    first.add(Conv2D(filters=16, input_shape=(F, K, 1), padding="same",kernel_initializer=initializers.glorot_normal(seed=123),kernel_size=(3, 3), activation='relu'))
    first.add(MaxPooling2D((2, 2), strides=(2, 2)))
    first.add(Conv2D(filters=16, padding="same",kernel_initializer=initializers.glorot_normal(seed=123), kernel_size=(8, 8),activation='relu'))
    first.add(MaxPooling2D((2, 2), strides=(2, 2)))
    first.add(Dropout(0.8, seed=12))
    first.add(Flatten())
    second = Sequential()
    second.add(Convolution1D(10, kernel_size = 2, kernel_initializer=initializers.glorot_normal(seed=123),input_shape = (C,1), activation='relu'))
    second.add(Bidirectional(LSTM(8,kernel_initializer=initializers.glorot_normal(seed=123))))
    model = Sequential()
    model.add(Merge([first, second], mode='concat'))
    model.add(Dense(T,kernel_initializer=initializers.glorot_normal(seed=123), activation = 'softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

#suppose word_embedding _matrix n * f * k, feature_matrix n* c, label with one-hot encoding n*t
#Y_train = keras.utils.to_categorical(label1, num_classes=6) for one-hot encoding

def run(m_train, m_test, feature_train, feature_test, y_train, y_test):
    batch_size = 10
    nb_epoch = 50
    N, F, K = m_train.shape
    m_train = m_train.reshape(N,F,K,1)
    m_test = m_test.reshape(m_test.shape + (1,))
    C = feature_train.shape[1]
    T = y_train.shape[1]
    feature_train = feature_train.reshape(N, C, 1)
    feature_test = feature_test.reshape(feature_test.shape+(1,))
    model = create_model(F, K, C, T)
    callbacks = [EarlyStopping(monitor='val_loss', patience=nb_epoch, verbose=0)]
    history = model.fit([m_train, feature_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, \
                        shuffle=True, verbose=2, validation_data=([m_test, feature_test], y_test), \
                        callbacks=callbacks)  # record history by JH
    model.save('hcnn_glorot_0.01_10.h5')
    return (model.predict([m_test, feature_test], batch_size=batch_size, verbose=2))
	

tmp = run(train_embedding, test_embedding,feature_train, feature_test, Y_train, Y_test)