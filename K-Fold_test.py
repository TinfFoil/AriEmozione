from Simple_NN_Test_2 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# These are the best epochs combined with the best neurons


def kFold_evaluation(combinations, train, train_y, test, test_y): #this performs k-fold and evaluates
    acc_per_fold = []
    loss_per_fold = []
    reports = []
    cf_matrices = []

    num_folds = 10
    kfold = KFold(n_splits=num_folds)
    inputs = np.concatenate((train, test), axis=0)
    targets = np.concatenate((train_y, test_y), axis=0)
    if len(combinations) == 1:
        for i in combinations.keys():
            a, b, c, d = i
            # K-fold Cross Validation model evaluation
            fold_no = 1
            for train, test in kfold.split(inputs, targets):
                model = Sequential()
                model.add(Dense(a, input_dim=dimensions, activation='relu'))
                model.add(Dense(b))
                model.add(Dropout(0.2))
                model.add(Activation('relu'))
                model.add(Dense(c))
                model.add(Dropout(0.2))
                model.add(Activation('relu'))
                model.add(Dense(6, activation="softmax"))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')
                model.fit(inputs[train], targets[train], nb_epoch=d, batch_size=32, verbose=1)

                # Generate generalization metrics
                scores = model.evaluate(inputs[test], targets[test], verbose=0)
                print(
                    f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])

                Y = np.argmax(targets[test], axis=1)  # here we build the precision/recall/f1 table for each fold
                y_pred = model.predict_classes(inputs[test])
                reports.append(classification_report(Y, y_pred))
                cm = confusion_matrix(Y, y_pred)
                cm = cm / cm.astype(np.float).sum(axis=1)
                cf_matrices.append(cm.round(2))
                # Increase fold number
                fold_no += 1

            # == Provide average scores ==
            print('------------------------------------------------------------------------')
            print(i)
            print('Score per fold')
            for n in range(0, len(acc_per_fold)):
                print('------------------------------------------------------------------------')
                print(f'> Fold {n + 1} - Loss: {loss_per_fold[n]} - Accuracy: {acc_per_fold[n]}%')
                print('------------------------------------------------------------------------')
                print(f'> Per Class Report:\n{reports[n]}')
            print('------------------------------------------------------------------------')
            print('Average scores for all folds:')
            print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
            print(f'> Loss: {np.mean(loss_per_fold)}')
            print(f'> Confusion Matrix:\n{np.nanmean(cf_matrices, axis=0)}')
            print('------------------------------------------------------------------------')
            model_structure = model.to_json()
            with open(f"Word_NN_ldia_{i}.json", "w") as json_file:
                json_file.write(model_structure)  # saves just the architecture
            model.save_weights(f"Word_Weights_ldia_{i}")  # saves the weights

    else:
        for (neurons, val) in combinations.items():
            one_combo = dict()
            one_combo[neurons] = val
            kFold_evaluation(one_combo, train, train_y, test, test_y)


kFold_evaluation(best_combinations, x_train, x_test, y_train, y_test)
