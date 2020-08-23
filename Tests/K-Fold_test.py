from Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

# These are the best epochs combined with the best neurons
best_neuron_x_epoch_3svd = {(32, 256, 256, 5): 0.40400001406669617, (32, 256, 256, 6): 0.4000000059604645, (32, 256, 256, 8): 0.4000000059604645, (32, 256, 256, 10): 0.4000000059604645, (32, 256, 256, 11): 0.40799999237060547, (32, 256, 256, 12): 0.4000000059604645, (64, 64, 32, 4): 0.41999998688697815, (64, 256, 128, 3): 0.4000000059604645, (64, 256, 128, 5): 0.4000000059604645, (64, 256, 128, 6): 0.40400001406669617, (96, 64, 256, 4): 0.41600000858306885, (96, 64, 256, 5): 0.4399999976158142, (96, 64, 256, 6): 0.42399999499320984, (96, 64, 256, 7): 0.4359999895095825, (96, 64, 256, 8): 0.42800000309944153, (96, 64, 256, 9): 0.42399999499320984, (96, 64, 256, 10): 0.41200000047683716, (96, 64, 256, 11): 0.42399999499320984, (96, 256, 96, 4): 0.4000000059604645, (96, 256, 96, 8): 0.40400001406669617, (96, 256, 96, 9): 0.40799999237060547, (128, 64, 64, 7): 0.40799999237060547, (128, 64, 64, 9): 0.41999998688697815, (128, 64, 64, 11): 0.4000000059604645, (128, 64, 96, 3): 0.41600000858306885, (128, 64, 96, 4): 0.41600000858306885, (128, 64, 96, 6): 0.40400001406669617, (128, 64, 96, 7): 0.41600000858306885, (128, 64, 96, 10): 0.40799999237060547, (128, 64, 256, 4): 0.41999998688697815, (128, 64, 256, 7): 0.40400001406669617, (128, 96, 128, 4): 0.40799999237060547, (128, 96, 128, 5): 0.41200000047683716, (128, 96, 128, 6): 0.4000000059604645}
print(len(best_neuron_x_epoch_3svd ))
best_neuronsepoch_3svd = dict()
for x in best_neuron_x_epoch_3svd.keys():
    if best_neuron_x_epoch_3svd[x] > 0.415:
        best_neuronsepoch_3svd[x] = best_neuron_x_epoch_3svd[x]
print(best_neuronsepoch_3svd)

best_parameters = {(96, 64, 256, 7): 0.4359999895095825}
#(96, 64, 256, 7): 0.4359999895095825, (96, 64, 256, 5): 0.4399999976158142, (128, 64, 96, 7): 0.41600000858306885

acc_per_fold = []
loss_per_fold = []
reports = []

num_folds = 10
kfold = KFold(n_splits=num_folds)
inputs = np.concatenate((trigram_svd, trigram_svd_dev), axis=0)
targets = np.concatenate((dummy_y, dummy_dev), axis=0)
for i in best_parameters.keys():
    a, b, c, d = i
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        model = Sequential()
        model.add(Dense(a, input_dim=32, activation='relu'))
        model.add(Dense(b))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(c))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(7, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        history = model.fit(inputs[train], targets[train], nb_epoch=d, batch_size=32, verbose=1)

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        Y = np.argmax(targets[test], axis=1) # here we build the precision/recall/f1 table for each fold
        y_pred = model.predict_classes(inputs[test])
        reports.append(classification_report(Y, y_pred))

        # Increase fold number
        fold_no += 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print(i)
    print('Score per fold')
    for n in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {n + 1} - Loss: {loss_per_fold[n]} - Accuracy: {acc_per_fold[n]}%')
        print(f'> Per Class Report:\n{reports[n]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    # model_structure = model.to_json()
    #  with open("NN_(96, 64, 256, 7).json", "w") as json_file:
    #   json_file.write(model_structure)  # saves just the architecture
    # model.save_weights(f"Weights_(96, 64, 256, 7)")  # saves the weights
