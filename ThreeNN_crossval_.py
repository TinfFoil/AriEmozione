from Vectorization import *
#If one wants to rerun the whole pipeline to find other Neurons*Epochs this should be changed to:
#from Finding_the_best_NeuronCombos import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def ThreeNN_crossval(combinations, tra, tra_y, tes, tes_y, dims):  # this performs k-fold and evaluates
    acc_per_fold = []
    loss_per_fold = []
    reports = []
    cf_matrices = []

    num_folds = 10
    kfold = KFold(n_splits=num_folds)
    inputs = np.concatenate((tra, tes), axis=0)
    targets = np.concatenate((tra_y, tes_y), axis=0)
    if len(combinations) == 1:
        for i in combinations.keys():
            a, b, c, d = i
            # K-fold Cross Validation model evaluation
            fold_no = 1
            for train, test in kfold.split(inputs, targets):
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
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')
                model.fit(inputs[train], targets[train], epochs=d, batch_size=32, verbose=0)

                # Generate generalization metrics
                scores = model.evaluate(inputs[test], targets[test], verbose=0)
                print(
                    f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])

                Y = np.argmax(targets[test], axis=1)  # here we build the precision/recall/f1 table for each fold
                y_pred = model.predict_classes(inputs[test])
                reports.append(classification_report(Y, y_pred, output_dict=True))
                cm = confusion_matrix(Y, y_pred)
                cm = cm / cm.astype(np.float).sum(axis=1)
                cf_matrices.append(cm.round(2))
                # Increase fold number
                fold_no += 1

            # == Provide average scores ==
            print('------------------------------------------------------------------------')

            print('Score per fold')
            f1_report = []
            for n in range(0, len(acc_per_fold)):
                print('------------------------------------------------------------------------')
                print(f'> Fold {n + 1} - Loss: {loss_per_fold[n]} - Accuracy: {acc_per_fold[n]}%')
                print('------------------------------------------------------------------------')
                print(f'> Per Class Report:\n{reports[n]}')
                f1_report.append(reports[n]['weighted avg']['f1-score'])
            print('------------------------------------------------------------------------')
            print(i)
            print('Average scores for all folds:')
            print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
            print(f'> F1-Score: {np.mean(f1_report).round(3)} (+-{np.std(f1_report).round(3)})')
            print(f'> Loss: {np.mean(loss_per_fold)}')
            print(f'> Confusion Matrix:\n{np.nanmean(cf_matrices, axis=0)}')
            print('------------------------------------------------------------------------')
            '''to be added if one wants to save the model:
            model_structure = model.to_json()
            with open(f"NN_tfidf_{i}.json", "w") as json_file:
               json_file.write(model_structure) 
            model.save_weights(f"Weights_tfidf_{i}")  
            '''
    else:
        for (neurons, val) in combinations.items():
            one_combo = dict()
            one_combo[neurons] = val
            ThreeNN_crossval(one_combo, tra, tra_y, tes, tes_y, dims)

# These are the Neuron*Epochs obtained from the Finding_the_best_NeuronCombos.py - the values are set to 1 since
# they are just placeholders

# 3-gram best Neurons*Epochs
tfidf_dict = {(64, 96, 128, 3): 1, (96, 32, 96, 3): 1, (128, 96, 256, 3): 1}
svd_dict = {(96, 96, 96, 10): 1, (96, 256, 64, 14): 1, (128, 128, 96, 10): 1}
ldia_dict = {(96, 256, 64, 8): 1, (128, 64, 96, 13): 1, (128, 96, 96, 7): 1}
clean_tfidf_dict = {(128, 96, 256, 2): 1, (128, 256, 64, 2): 1, (96, 96, 96, 4): 1}
clean_svd_dict = {(128, 63, 32, 8): 1, (128, 32, 64, 6): 1, (128, 32, 64, 5): 1}
clean_ldia_dict= {(128, 64, 256, 11): 1, (128, 96, 128, 3): 1, (32, 96, 32, 9): 1}

# Word best Meurons*Epochs
word_tfidf_dict = {(128, 32, 128, 10): 1, (96, 64, 128, 3): 1, (96, 64, 128, 6): 1}
word_svd_dict = {(128, 96, 64, 7): 1, (128, 256, 128, 5): 1, (64, 256, 64, 12): 1}
word_ldia_dict = {(32, 96, 256, 1): 1, (96, 96, 256, 1): 1, (32, 96, 128, 14): 1}


print("char 3-grams")
ThreeNN_crossval(tfidf_dict, trigram_tfidf_cv, cv_y, trigram_tfidf_dev, dev_y, dim_cv_char)
ThreeNN_crossval(word_tfidf_dict, trigram_tfidf_cv, cv_y, trigram_tfidf_dev, dev_y, dim_cv_char)
ThreeNN_crossval(clean_tfidf_dict, trigram_tfidf_cv, cv_y, trigram_tfidf_dev, dev_y, dim_cv_char)

print("words")
ThreeNN_crossval(tfidf_dict, cv_tfidf, cv_y, dev_tfidf, dev_y, dim_cv_word)
ThreeNN_crossval(word_tfidf_dict, cv_tfidf, cv_y, dev_tfidf, dev_y, dim_cv_word)

print("LDA char 3-grams")
ThreeNN_crossval(ldia_dict, trigram_ldia_cv, cv_y, trigram_ldia_dev, dev_y, 32)
ThreeNN_crossval(word_ldia_dict, trigram_ldia_cv, cv_y, trigram_ldia_dev, dev_y, 32)
ThreeNN_crossval(clean_ldia_dict, trigram_ldia_cv, cv_y, trigram_ldia_dev, dev_y, 32)

print("LSA char 3-grams")
ThreeNN_crossval(svd_dict, trigram_svd_cv, cv_y, trigram_svd_dev, dev_y, 32)
ThreeNN_crossval(word_svd_dict, trigram_svd_cv, cv_y, trigram_svd_dev, dev_y, 32)
ThreeNN_crossval(clean_svd_dict, trigram_svd_cv, cv_y, trigram_svd_dev, dev_y, 32)

print("LDA words")
ThreeNN_crossval(ldia_dict, cv_ldia, cv_y, dev_ldia, dev_y, 32)
ThreeNN_crossval(word_ldia_dict, cv_ldia, cv_y, dev_ldia, dev_y, 32)

print("LSA words")
ThreeNN_crossval(svd_dict, cv_svd, cv_y, dev_svd, dev_y, 32)
ThreeNN_crossval(word_svd_dict, cv_svd, cv_y, dev_svd, dev_y, 32)


'''
If one wants to rerun the whole pipeline to find other Neurons*Epochs, comment out the code above and include
the following code:

ThreeNN_crossval(best_combinations, x_train, x_test, y_train, y_test, dm)
'''





