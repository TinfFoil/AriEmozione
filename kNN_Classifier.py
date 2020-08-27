from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Tokenize_Vectorize import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold


def kNN_KFOLD(train, train_y, test, test_y):
    acc_per_fold = []
    reports = []
    cf_matrices = []

    kfold = KFold(n_splits=10)
    inputs = np.concatenate((train, test), axis=0)
    targets = np.concatenate((train_y, test_y), axis=0)
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        kNN = KNeighborsClassifier(n_neighbors=1)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        kNN = kNN.fit(inputs[train], targets[train])
        y_pred = kNN.predict(inputs[test])
        acc = metrics.accuracy_score(targets[test], y_pred)
        acc_per_fold.append(acc * 100)
        print(f"kNN Accuracy per Fold {fold_no} with k=1:", acc)

        reports.append(classification_report(targets[test], y_pred))

        cm = confusion_matrix(targets[test].argmax(axis=1), y_pred.argmax(axis=1))
        cm = cm / cm.astype(np.float).sum(axis=1)
        cf_matrices.append(cm.round(2))
        print(f"kNN Confusion Matrix with k=1:\n", cm.round(2))


        fold_no += 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for n in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {n + 1} - Accuracy: {acc_per_fold[n]}%')
        print('------------------------------------------------------------------------')
        # print(f'> Per Class Report:\n{reports[n]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Confusion Matrix:\n{np.nanmean(cf_matrices, axis=0)}')
    print('------------------------------------------------------------------------')


kNN_KFOLD(aria_svd, dummy_y, dev_svd, dummy_dev)
kNN_KFOLD(aria_tfidf, dummy_y, dev_tfidf, dummy_dev)
kNN_KFOLD(aria_ldia, dummy_y, dev_ldia, dummy_dev)
kNN_KFOLD(trigram_tfidf, dummy_y, trigram_tfidf_dev, dummy_dev)
kNN_KFOLD(trigram_svd, dummy_y, trigram_svd_dev, dummy_dev)
kNN_KFOLD(trigram_ldia, dummy_y, trigram_ldia_dev, dummy_dev)
kNN_KFOLD(clean_tfidf, dummy_y, clean_tfidf_dev, dummy_dev)
kNN_KFOLD(clean_svd, dummy_y, clean_svd_dev, dummy_dev)
kNN_KFOLD(clean_ldia, dummy_y, clean_ldia_dev, dummy_dev)
