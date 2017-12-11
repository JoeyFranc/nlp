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



#X_train1 = np.random.random((200, 100, 80,1))
#X_valid1 = np.random.random((100, 100, 80,1))
#X_train2 = np.random.random((200, 5,1))
#X_valid2 = np.random.random((100, 5,1))
#label1 = np.random.randint(6, size=(200, 1))
#label2 = np.random.randint(6, size=(100, 1))

# Convert labels to categorical one-hot encoding

#Y_valid = keras.utils.to_categorical(label2, num_classes=6)
#sgd = SGD(lr=0.9, decay=1e-6, momentum=0.9, nesterov=True)

def create_model(F, K, T):
	first = Sequential()
	first.add(Conv2D(filters = 128, input_shape=(F,K,1), padding="same",kernel_initializer=initializers.glorot_normal(seed=123),kernel_size = (2, 2), activation = 'relu'))
	first.add(BatchNormalization())
	first.add(MaxPooling2D((2, 2), strides=(2,2)))
	first.add(Conv2D(filters=128, padding="same",kernel_initializer=initializers.glorot_normal(seed=94),kernel_size=(3, 3), activation='relu'))
	first.add(BatchNormalization())
	first.add(MaxPooling2D((2, 2), strides=(2, 2)))
	first.add(Conv2D(filters=128,padding="same", kernel_initializer=initializers.glorot_normal(seed=213), kernel_size=(4, 4),activation='relu'))
	first.add(BatchNormalization())
	first.add(MaxPooling2D((2, 2), strides=(2, 2)))
	first.add(Dropout(0.8, seed = 12))
	first.add(Flatten())
	first.add(Dense(T, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=12)))
	sgd =  SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	first.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
	return first

#suppose word_embedding _matrix n * f * k, feature_matrix n* c, label with one-hot encoding n*t
#Y_train = keras.utils.to_categorical(label1, num_classes=6) for one-hot encoding


def run_cnn(m_train, m_test, y_train, y_test):
	batch_size = 50
	nb_epoch = 10
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
	model.save('glorot_0.01_128.h5')
	return (model.predict(m_test, batch_size=batch_size, verbose=2))


def run_classifier(X_train, X_test, y_train, y_test):
    train_embeddings, _ = X_train
    test_embeddings, _ = X_test
    return run_cnn(train_embeddings, test_embeddings, y_train, y_test)

