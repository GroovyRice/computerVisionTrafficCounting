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

# INTEGER TENSOR
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# MODEL DEFINITION
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))  # Softmax outputs a probability distribution summation to 1

# COMPILING THE MODEL
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

# VALIDATION SET
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# TRAINING THE MODEL
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

# EVALUATE THE MODEL
final = model.evaluate(x_test, y_test)
print("=" * 100)
print("Test Loss:", final[0], "  |  Test Accuracy:", final[1] * 100, "%")
print("=" * 100)

# PREDICT THE MODEL
# prediction = model.predict(x_test)
