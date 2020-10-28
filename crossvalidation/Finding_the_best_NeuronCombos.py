from Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
import itertools

# Set the various combinations of neurons
input_neuron = [32, 64, 96, 128]
first_neuron = [32, 64, 96, 128, 256]
second_neuron = [32, 64, 96, 128, 256]
combinations = list(itertools.product(*[input_neuron, first_neuron, second_neuron]))

# Set the representation to be tested
x_train = cv_tfidf
x_test = cv_y
y_train = dev_tfidf
y_test = dev_y
dm = dim_cv_word  # This is the dimension of the vectors, to be found in Vectorization.py


def try_three_layers(tup, tra, tra_y, tes, tes_y, dims):
    combination = list()
    for i in tup:
        model = Sequential()
        model.add(Dense(i[0], input_dim=dims, activation='relu'))
        model.add(Dense(i[1]))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(i[2]))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(tra, tra_y, nb_epoch=15, batch_size=32, verbose=1)
        scores = model.evaluate(tes, tes_y, verbose=0)
        '''print("the number of neurons",a,b,c)'''
        print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
        append_this = (i, scores[1])
        combination.append(append_this)

    combination = combination.sort(key=lambda x: x[1])
    combination = combination[-20:]
    dictionary_three_layers = dict()
    for (neurons, score) in combination:
        dictionary_three_layers[neurons] = score
    return dictionary_three_layers


def three_layers_try_epoch(combination_dict, tra, tra_y, tes, tes_y, dims):
    d = 0
    new_combos = list()
    for i in combination_dict.keys():
        a, b, c = i
        model = Sequential()
        model.add(Dense(a, input_dim=dims, activation='relu'))
        model.add(Dense(b))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(c))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        for i in range(15):
            d = i
            model.fit(tra, tra_y, nb_epoch=d, batch_size=32, verbose=1)
            score = model.evaluate(tes, tes_y, verbose=0)
            print(f"The number of neurons: {a, b, c}\nThe number of epoch: {d}")
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
            append_this = ((a, b, c, d), score[1])
            new_combos.append(append_this)

    new_combos = new_combos.sort(key=lambda x: x[1])
    new_combos = new_combos[-3:]
    dictionary_three_layers_epoch = dict()
    for (neurons_epochs, sco) in new_combos:
        dictionary_three_layers_epoch[neurons_epochs] = sco
    return dictionary_three_layers_epoch


def neuron_x_epoch_selection(tup, train, train_y, test, test_y, dims):
    best_neurons_ = try_three_layers(tup, train, train_y, test, test_y, dims)
    best_neurons_x_epoch = three_layers_try_epoch(best_neurons_, train, train_y, test, test_y, dims)
    return best_neurons_x_epoch, best_neurons_


best_combinations, best_neurons = neuron_x_epoch_selection(combinations, x_train, x_test, y_train, y_test, dm)

if __name__ == '__main__':
    print('The best neurons: ', best_neurons)
    print('The best neurons*epochs: ', best_combinations)
