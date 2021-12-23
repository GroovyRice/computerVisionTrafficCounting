from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(max([max(sequence) for sequence in train_data]))
# Finds the Max value in each List of Train Data and concatenates it into an array
# the maximum value is then found and print from ALL lists max(train_data[0])
# prints the maximum value in one list which is 7486. Then if each lists maximum is found
# via max(train_data[1]) till train_data[len(train_data)-1] then whatever the
# maximum of all those values is the maximum word value.
# This is essentially what is occurring:
# temp2 = 0
# for sequence in train_data:
#     temp1 = max(sequence)
#     if temp1 > temp2:
#         temp2 = temp1
# print(temp2)
#
# Or to further this it can occur with an array like so
# print([5 * i for i in [1, 2, 3, 4, 5]])
# which will print 5, 10, 15, 20, 25
