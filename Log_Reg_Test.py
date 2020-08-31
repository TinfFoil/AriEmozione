from Tokenize_Vectorize import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

def LogReg_KFOLD(train, train_y, test, test_y):
    acc_per_fold = []
    reports = []
    cf_matrices = []

    kfold = KFold(n_splits=10)
    inputs = np.concatenate((train, test), axis=0)
    targets = np.concatenate((train_y, test_y), axis=0)
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        LogReg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        LogReg = LogReg.fit(inputs[train], targets[train])
        y_pred = LogReg.predict(inputs[test])
        acc = metrics.accuracy_score(targets[test], y_pred)
        acc_per_fold.append(acc * 100)
        print(f"LogReg Accuracy per Fold {fold_no} with k=1:", acc)
        reports.append(classification_report(targets[test], y_pred, output_dict=True))
        cm = confusion_matrix(targets[test], y_pred)
        cm = cm / cm.astype(np.float).sum(axis=1)
        cf_matrices.append(cm.round(2))
        print(f"Logistic Regression Confusion Matrix\n", cm.round(2))
        fold_no += 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    f1_report = []
    for n in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {n + 1} - Accuracy: {acc_per_fold[n]}%')
        print('------------------------------------------------------------------------')
        print(f'> Per Class Report:\n{reports[n]}')
        f1_report.append(reports[n]['weighted avg']['f1-score'])
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold).round(3)} (+- {np.std(acc_per_fold).round(3)})')
    print(f'> F1-Score: {np.mean(f1_report).round(3)} (+-{np.std(f1_report).round(3)})')
    print(f'> Confusion Matrix:\n{np.nanmean(cf_matrices, axis=0)}')
    print('------------------------------------------------------------------------')

LogReg_KFOLD(aria_svd, encoded_train, dev_svd, encoded_dev)
LogReg_KFOLD(aria_tfidf, encoded_dev, dev_tfidf, encoded_dev)
LogReg_KFOLD(aria_ldia, encoded_dev, dev_ldia, encoded_dev)
LogReg_KFOLD(trigram_tfidf, encoded_dev, trigram_tfidf_dev, encoded_dev)
LogReg_KFOLD(trigram_svd, encoded_dev, trigram_svd_dev, encoded_dev)
LogReg_KFOLD(trigram_ldia, encoded_dev, trigram_ldia_dev, encoded_dev)
LogReg_KFOLD(clean_tfidf, encoded_dev, clean_tfidf_dev, encoded_dev)
LogReg_KFOLD(clean_svd, encoded_dev, clean_svd_dev, encoded_dev)
LogReg_KFOLD(clean_ldia, encoded_dev, clean_ldia_dev, encoded_dev)