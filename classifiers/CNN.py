'''
    CNN.py
    @author Jiahua Chen
    
    The pure CNN architecture.
'''



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
from word_embeddings import train_embedding, test_embedding, Y_train, Y_test



def create_model(F, K, T):
	first = Sequential()
	first.add(Conv2D(filters = 64, input_shape=(F,K,1), padding="same",kernel_initializer=initializers.he_normal(seed=315),kernel_size = (2, 2), activation = 'relu'))
	first.add(BatchNormalization())
	first.add(MaxPooling2D((2, 2), strides=(2,2)))
	first.add(Conv2D(filters=64, padding="same",kernel_initializer=initializers.he_normal(seed=944),kernel_size=(3, 3), activation='relu'))
	first.add(BatchNormalization())
	first.add(MaxPooling2D((2, 2), strides=(2, 2)))
	first.add(Conv2D(filters=64,padding="same", kernel_initializer=initializers.he_normal(seed=984), kernel_size=(4, 4),activation='relu'))
	first.add(BatchNormalization())
	first.add(MaxPooling2D((2, 2), strides=(2, 2)))
	first.add(Dropout(0.8, seed = 12))
	first.add(Flatten())
	first.add(Dense(T, activation='softmax', kernel_initializer=initializers.he_normal(seed=1042)))
	sgd =  SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	first.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
	return first

#suppose word_embedding _matrix n * f * k, feature_matrix n* c, label with one-hot encoding n*t
#Y_train = keras.utils.to_categorical(label1, num_classes=6) for one-hot encoding


def run(m_train, m_test, y_train, y_test):
	batch_size = 50
	nb_epoch = 70
	N, F, K = m_train.shape
	T = y_train.shape[1]
	m_train = m_train.reshape(N, F, K, 1)
	m_test = m_test.reshape(m_test.shape + (1,))
	#m_valid = m_valid.reshape(m_valid.shape + (1,))
	model = create_model(F, K, T)
	callbacks = [EarlyStopping(monitor='val_loss', patience=nb_epoch, verbose=0)]
	model.fit(m_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, \
                        shuffle=True, verbose=2, validation_data=(m_test, y_test), \
                        callbacks=callbacks)  # record history by JH
	#model.save('he_0.01_64_50.h5')
	return (model.predict(m_test, batch_size=batch_size, verbose=2))


def run_classifier(X_train, X_test, y_train, y_test):
    train_embeddings, _ = X_train
    test_embeddings, _ = X_test
    return run_cnn(train_embeddings, test_embeddings, y_train, y_test)

