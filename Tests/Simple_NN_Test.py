from Tokenize_Vectorize import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation


# We ran several experiments and the following neuron combinations were the best for 3-grams SVD and simple 3-grams
results_trigram_svd = {(32, 32, 128): 0.3959999978542328, (32, 96, 256): 0.4000000059604645, (32, 128, 128): 0.41200000047683716, (32, 256, 256): 0.41999998688697815, (64, 32, 64): 0.3959999978542328, (64, 32, 96): 0.3919999897480011, (64, 64, 32): 0.4399999976158142, (64, 128, 32): 0.41200000047683716, (64, 128, 256): 0.40400001406669617, (64, 256, 32): 0.3919999897480011, (64, 256, 128): 0.42800000309944153, (64, 256, 256): 0.3919999897480011, (96, 32, 32): 0.41200000047683716, (96, 32, 96): 0.3959999978542328, (96, 32, 128): 0.4000000059604645, (96, 64, 96): 0.41200000047683716, (96, 64, 256): 0.41600000858306885, (96, 96, 96): 0.4000000059604645, (96, 96, 256): 0.41600000858306885, (96, 128, 32): 0.40799999237060547, (96, 128, 64): 0.3959999978542328, (96, 128, 96): 0.3919999897480011, (96, 128, 256): 0.3959999978542328, (96, 256, 32): 0.40400001406669617, (96, 256, 64): 0.4320000112056732, (96, 256, 96): 0.4519999921321869, (96, 256, 256): 0.3959999978542328, (128, 32, 96): 0.3959999978542328, (128, 64, 32): 0.40400001406669617, (128, 64, 64): 0.4320000112056732, (128, 64, 96): 0.41600000858306885, (128, 64, 128): 0.41200000047683716, (128, 64, 256): 0.42399999499320984, (128, 96, 96): 0.40400001406669617, (128, 96, 128): 0.41999998688697815, (128, 128, 32): 0.3919999897480011, (128, 128, 96): 0.40400001406669617, (128, 128, 128): 0.41200000047683716, (128, 128, 256): 0.40400001406669617, (128, 256, 32): 0.40799999237060547}
print(len(results_trigram_svd))
best_neurons_3svd = dict()
for x in results_trigram_svd.keys():
    if results_trigram_svd[x] > 0.415:
        best_neurons_3svd[x] = results_trigram_svd[x]
print(best_neurons_3svd)

results_trigram = {(32, 32, 64): 0.4000000059604645, (32, 32, 96): 0.3919999897480011, (32, 256, 32): 0.4000000059604645, (32, 256, 64): 0.42399999499320984, (64, 32, 64): 0.41999998688697815, (64, 32, 256): 0.4000000059604645, (64, 64, 64): 0.3959999978542328, (64, 64, 256): 0.3959999978542328, (64, 96, 128): 0.3919999897480011, (64, 128, 64): 0.3919999897480011, (96, 64, 32): 0.40400001406669617, (96, 64, 64): 0.40400001406669617, (96, 64, 96): 0.41200000047683716, (96, 64, 256): 0.41200000047683716, (96, 96, 96): 0.4000000059604645, (96, 128, 64): 0.40400001406669617, (96, 128, 256): 0.40400001406669617, (96, 256, 64): 0.3959999978542328, (96, 256, 96): 0.3959999978542328, (96, 256, 128): 0.3919999897480011, (128, 32, 64): 0.3919999897480011, (128, 32, 128): 0.3959999978542328, (128, 64, 32): 0.3959999978542328, (128, 64, 96): 0.3919999897480011, (128, 128, 256): 0.3959999978542328, (128, 256, 64): 0.40400001406669617}
print(len(results_trigram))
best_neurons_3 = dict()
for x in results_trigram.keys():
    if results_trigram[x] > 0.415:
        best_neurons_3[x] = results_trigram[x]
print(best_neurons_3)


# Here we find the best epochs

def three_layers_try_epoch(combination_dict):
    d = 0
    dictionary_three_layers = dict()
    for i in combination_dict.keys():
        a,b,c=i
        model=Sequential()
        model.add(Dense(a,input_dim=32, activation='relu'))
        model.add(Dense(b))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(c))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(7,activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        for i in range(15):
            d = i
            model.fit(trigram_svd, dummy_y, nb_epoch=d, batch_size=32, verbose=1)
            score = model.evaluate(trigram_svd_dev, dummy_dev, verbose=0)
            print(f"The number of neurons: {a,b,c}\nThe number of epoch: {d}")
            print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
            if score[1] >= 0.4:
                dictionary_three_layers[(a,b,c,d)] = score[1]
    return dictionary_three_layers


best_epoch_x_neurons = three_layers_try_epoch(best_neurons_3svd)
print(best_epoch_x_neurons)
