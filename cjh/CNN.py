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
#X_train1 = np.random.random((200, 100, 80,1))
#X_valid1 = np.random.random((100, 100, 80,1))
#X_train2 = np.random.random((200, 5,1))
#X_valid2 = np.random.random((100, 5,1))
#label1 = np.random.randint(6, size=(200, 1))
#label2 = np.random.randint(6, size=(100, 1))

# Convert labels to categorical one-hot encoding

#Y_valid = keras.utils.to_categorical(label2, num_classes=6)


def create_model(F, K, T):
	first = Sequential()
	first.add(Conv2D(filters = 128, input_shape=(F,K,1), kernel_initializer=initializers.RandomNormal(stddev=0.01),kernel_size = (2, 2), activation = 'relu'))
	first.add(MaxPooling2D((2, 2), strides=(2,2)))
	first.add(Dropout(0.8))
	first.add(Conv2D(filters=128, kernel_initializer=initializers.RandomNormal(stddev=0.01),kernel_size=(3, 3), activation='relu'))
	first.add(MaxPooling2D((2, 2), strides=(2, 2)))
	first.add(Dropout(0.8))
	first.add(Conv2D(filters=128, kernel_initializer=initializers.RandomNormal(stddev=0.01), kernel_size=(4, 4),activation='relu'))
	first.add(MaxPooling2D((2, 2), strides=(2, 2)))
	first.add(Dropout(0.8))
	first.add(Flatten())
	first.add(Dense(T, activation = 'softmax',kernel_initializer=initializers.RandomNormal(stddev=0.02),))
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	first.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
	return first

#suppose word_embedding _matrix n * f * k, feature_matrix n* c, label with one-hot encoding n*t
#Y_train = keras.utils.to_categorical(label1, num_classes=6) for one-hot encoding

def run(m_train, m_valid, y_train, y_valid):
	batch_size = 50
	nb_epoch = 5
	N, F, K = m_train.shape
	T = y_train.shape[1]
	m_train = m_train.reshape(N, F, K, 1)
	m_test = m_test.reshape(m_test.shape + (1,))
	m_valid = m_valid.reshape(m_valid.shape + (1,))
	model = create_model(F, K, T)
	callbacks = [EarlyStopping(monitor='val_loss', patience=nb_epoch, verbose=0)]
	history = model.fit(m_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, \
                        shuffle=True, verbose=2, validation_data=(m_valid, y_valid), \
                        callbacks=callbacks)  # record history by JH
	return (model.predict(m_test, batch_size=batch_size, verbose=2))
	

tmp = cnn(train_embedding, test_embedding, valid_embedding, Y_train, Y_valid)