from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Tokenize_Vectorize import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np


def kNN_crossval(train, tra_y, test, tes_y):
    acc_per_fold_k = []
    reports = []
    cf_matrices_k = []

    kfold = KFold(n_splits=10)
    inputs = np.concatenate((train, test), axis=0)
    targets = np.concatenate((tra_y, tes_y), axis=0)
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        for k in range(10):
            if k != 0:
                kNN = KNeighborsClassifier(n_neighbors=k)
                #print('------------------------------------------------------------------------')
                #print(f'Training for fold {fold_no} ...')
                kNN = kNN.fit(inputs[train], targets[train])
                y_pred = kNN.predict(inputs[test])
                acc = metrics.accuracy_score(targets[test], y_pred)
                acc_per_fold_k.append((acc * 100, k))

                reports.append((classification_report(targets[test], y_pred, output_dict=True), k))

                cm = confusion_matrix(targets[test].argmax(axis=1), y_pred.argmax(axis=1))
                cm = cm / cm.astype(np.float).sum(axis=1)
                cf_matrices_k.append((cm.round(2), k))


        fold_no += 1

    print('------------------------------------------------------------------------')
    #print('Score per fold')
    f1_report_k = []
    for n in range(0, len(acc_per_fold_k)):
        f1_report_k.append((reports[n][0]['weighted avg']['f1-score'],reports[n][1]))

    acc_per_fold = []
    acc_per_fold_k = sorted(acc_per_fold_k, key=lambda x: x[1])
    for n in range(9):
        acc_per_fold.append([x[0] for x in acc_per_fold_k[:10]])
        del acc_per_fold_k[:10]

    f1_report = []
    f1_report_k = sorted(f1_report_k, key=lambda x: x[1])
    for n in range(9):
        f1_report.append([x[0] for x in f1_report_k[:10]])
        del f1_report_k[:10]

    cf_matrices = []
    cf_matrices_k = sorted(cf_matrices_k, key=lambda x: x[1])
    for n in range(9):
        cf_matrices.append([x[0] for x in cf_matrices_k[:10]])
        del cf_matrices_k[:10]

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    for idx, score in enumerate(acc_per_fold):
        print(f'With k={idx+1}')
        print(f'> Accuracy: {np.mean(score).round(3)} (+- {np.std(score).round(3)})')
        print(f'> F1-Score: {np.mean(f1_report[idx]).round(3)}')
        print(f'> Confusion Matrix:\n{np.nanmean(cf_matrices[idx], axis=0)}')
    print('------------------------------------------------------------------------')


print('Word TFIDF')
kNN_crossval(cv_tfidf, cv_y, dev_tfidf, dev_y)
print('Word SVD')
kNN_crossval(cv_svd, cv_y, dev_svd, dev_y)
print('Word LDiA')
kNN_crossval(cv_ldia, cv_y, dev_ldia, dev_y)
print('3-gram TFIDF')
kNN_crossval(trigram_tfidf_cv, cv_y, trigram_tfidf_dev, dev_y)
print('3-gram SVD')
kNN_crossval(trigram_svd_cv, cv_y, trigram_svd_dev, dev_y)
print('3-gram LDiA')
kNN_crossval(trigram_ldia_cv, cv_y, trigram_ldia_dev, dev_y)




