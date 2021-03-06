from keras.datasets import imdb
from keras import models, layers
import numpy as np

# LOADING IMDB DATASET
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# ENCODING THE INTEGER SEQUENCES INTO BINARY MATRIX
def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# VECTORISING DATA
x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)

# VECTORISING LABELS
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# DEFINING MODEL
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# COMPILING MODEL
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# DEFINING VALIDATION SET
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# TRAINING MODEL
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
# training_results = model.evaluate(x_test, y_test)
# print(training_results)

# PREDICTING MODEL ON NEW DATA
final = model.predict(x_test)
np.set_printoptions(precision=3)
print(final)

# NOTE: .evaluate and .predict can not be run within the same document
