from Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
import itertools

# build the model
input_neuron = [8, 16, 32, 64, 96]
first_neuron = [8, 16, 32, 64, 96]
combinations = list(itertools.product(*[input_neuron, first_neuron]))

# Set the representation to be tested
x_train = cv_tfidf
x_test = cv_y
y_train = dev_tfidf
y_test = dev_y
dm = dim_cv_word  # The dimension of the TFIDF vectors, to be found in Vectorization.py. Otherwise, set it to 32.


def try_two_layers(tup, train, train_y, test, test_y, dims):  # this finds the best combination of neurons
    combination = list()
    for i in tup:
        model = Sequential()
        model.add(Dense(i[0], input_dim=dims, activation='relu'))
        model.add(Dense(i[1]))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train, train_y, epochs=10, batch_size=32, verbose=0)
        scores = model.evaluate(test, test_y, verbose=0)
        print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
        append_this = (i, scores[1])
        combination.append(append_this)

    combination.sort(key=lambda x: x[1])
    combination = combination[-20:]
    dictionary_two_layers = dict()
    for (neurons, score) in combination:
        dictionary_two_layers[neurons] = score
    return dictionary_two_layers


def two_layers_try_epoch(combination_dict, train, train_y, test, test_y,
                         dims):  # this finds the best combination of epochs
    d = 0
    new_combos = list()
    for n in range(15):
        for i in combination_dict.keys():
            a, b = i
            d = n
            model = Sequential()
            model.add(Dense(a, input_dim=dims, activation='relu'))
            model.add(Dense(b))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))
            model.add(Dense(6, activation="softmax"))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(train, train_y, epochs=d, batch_size=32, verbose=1)
            score = model.evaluate(test, test_y, verbose=0)
            print(f"The number of neurons: {a, b}\nThe number of epoch: {d}")
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
            append_this = ((a, b, d), score[1])
            new_combos.append(append_this)

    new_combos.sort(key=lambda x: x[1])
    new_combos = new_combos[-3:]
    dictionary_two_layers_epoch = dict()
    for (neurons_epochs, sco) in new_combos:
        dictionary_two_layers_epoch[neurons_epochs] = sco
    return dictionary_two_layers_epoch


def neuron_x_epoch_selection(tup, train, train_y, test, test_y, dims):
    best_neurons_ = try_two_layers(tup, train, train_y, test, test_y, dims)
    best_neurons_x_epoch = two_layers_try_epoch(best_neurons_, train, train_y, test, test_y, dims)
    return best_neurons_x_epoch, best_neurons_


best_combinations, best_neurons = neuron_x_epoch_selection(combinations, x_train, x_test, y_train, y_test, dm)

if __name__ == '__main__':
    print('The best neurons: ', best_neurons)
    print('The best neurons*epochs: ', best_combinations)
