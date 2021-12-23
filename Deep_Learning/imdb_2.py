from keras.datasets import imdb
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Maximum value of word:", max([max(sequence) for sequence in train_data]))


def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Creates temporary array of shape (len(sequences), dimension)
    # This for loop acts is an enumerating rather than collection based this means that 'i' will increase each iteration
    # Like with C programming whereas sequence will still be the current value of sequences at the time.
    # For example:
    # a = ["foo", "bar", "qua"]
    # for i, current in enumerate(a):
    #     print(i, current)
    # This is print out the following
    # 0 foo
    # 1 bar
    # 2 qua
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        # Each list is cycled through and every value is replaced in the zeros array with a 1 at there position For
        # example, if the current sequence is [3,5] and the results array is [0,0,0,0,0,0] the ones would be
        # attributed to the following [0,0,0,1,0,1] aka positions 3 and 5 in the matrix this is because results[1,[3,
        # 5]] is first row column 3 and 5 only thus only 3 and 5 would be changed to ones. To further this
        # understanding, say I have a matrix M which has the following [0,0,0,0] and I want to change column 1 and 3
        # to 3's for instance. This can be done with the following code M[0,[1,3]] = 3. Try it!
        # REMEMBER column 1 is the second from the left as matrix notation starts at 0!!!
    return results


x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)

# Vectorise the label data
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# Activation 'relu' stands for The Rectified Linear Unit Function which simply put if negative zeros, if positive
# keeps the same. The 'sigmoid' squashes arbitrary values into [0,1] this gives the output interpreted as a probability
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, (len(history_dict['acc']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
