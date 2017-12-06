# Imports
import time
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


np.random.seed(42)

# Loading the data (it's preloaded in Keras)
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                     num_words=None,
                                                     skip_top=2,
                                                     maxlen=None,
                                                     seed=113,
                                                     start_char=1,
                                                     oov_char=2,
                                                     index_from=3)

print(x_train.shape)
print(x_test.shape)

print(x_train[0])
print(y_train[0])

# at this point each number in inout is th eindex of hte word but we will one-hot it into
# 1,0 with length of vector being 1000 for top 100 words and each reviw having 1 if the word is there
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

start = time.time()


# Building the model
model = Sequential()
model.add(Dense(10, activation='linear', input_shape=(1000,)))
#model.add(Dropout(.2))
model.add(Dense(2, activation='sigmoid'))

# Compiling the model


model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

# Training the model
#PERFECT: model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=0)

model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=0)

'''
#DLNF solution was this woth carp accuracy 0.48

# Building the model architecture with one layer of length 100
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
'''

elapsed_time = float(time.time() - start)            
print("\rElapsedTime:", elapsed_time)

score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])