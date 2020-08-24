from Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
import itertools

# build the model
input_neuron = [32, 64, 96, 128]
first_neuron = [32, 64, 96, 128, 256]
second_neuron = [32, 64, 96, 128, 256]
combinations = list(itertools.product(*[input_neuron, first_neuron, second_neuron]))
x_train = trigram_svd
x_test = dummy_y
y_train = trigram_svd_dev
y_test = dummy_dev

def try_three_layers(tup, train, train_y, test, test_y): #this finds the best combination of neurons
    combinations = list()
    for i in tup:
        model = Sequential()
        model.add(Dense(i[0], input_dim=32, activation='relu'))
        model.add(Dense(i[1]))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(i[2]))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train, train_y, nb_epoch=13, batch_size=32, verbose=0)
        scores = model.evaluate(test, test_y, verbose=0)
        '''print("the number of neurons",a,b,c)'''
        print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
        append_this = (i, scores[1])
        combinations.append(append_this)

    combinations.sort(key=lambda x: x[1])
    combinations = combinations[-20:]
    dictionary_three_layers = dict()
    for (neurons, score) in combinations:
        dictionary_three_layers[neurons] = score
    return dictionary_three_layers


def three_layers_try_epoch(combination_dict, train, train_y, test, test_y): #this finds the best combination of epochs
    d = 0
    new_combos = list()
    for n in range(15):
        for i in combination_dict.keys():
            a, b, c = i
            d = n
            model = Sequential()
            model.add(Dense(a, input_dim=32, activation='relu'))
            model.add(Dense(b))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))
            model.add(Dense(c))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))
            model.add(Dense(6, activation="softmax"))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(train, train_y, nb_epoch=d, batch_size=32, verbose=1)
            score = model.evaluate(test, test_y, verbose=0)
            print(f"The number of neurons: {a, b, c}\nThe number of epoch: {d}")
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
            append_this = ((a, b, c, d), score[1])
            new_combos.append(append_this)

    new_combos.sort(key=lambda x: x[1])
    new_combos = new_combos[-3:]
    dictionary_three_layers_epoch = dict()
    for (neurons_epochs, sc) in new_combos:
        dictionary_three_layers_epoch[neurons_epochs] = sc
    return dictionary_three_layers_epoch


def neuron_x_epoch_selection(tup, train, train_y, test, test_y):
    best_neurons = try_three_layers(tup, train, train_y, test, test_y)
    best_neurons_x_epoch = three_layers_try_epoch(best_neurons, train, train_y, test, test_y)
    return best_neurons_x_epoch, best_neurons


best_combinations, best_neurons = neuron_x_epoch_selection(combinations, x_train, x_test, y_train, y_test)
print('The best neurons: ', best_neurons)
print('The best neurons*epochs: ', best_combinations)
