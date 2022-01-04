from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models, layers

# LOADING REUTERS DATASET
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# ENCODING THE DATA
def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)

# CATEGORICAL ENCODING
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# MODEL DEFINITION
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))  # Softmax outputs a probability distribution summation to 1

# COMPILING THE MODEL
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# VALIDATION SET
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# TRAINING THE MODEL
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

# EVALUATE THE MODEL
results = model.evaluate(x_test, one_hot_test_labels)
print("=" * 100)
print("Test Loss:", results[0], "  |  Test Accuracy:", results[1] * 100, "%")
print("=" * 100)

# PREDICT THE MODEL
# prediction = model.predict(x_test)
