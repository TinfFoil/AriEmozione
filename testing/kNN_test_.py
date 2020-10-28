from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Preprocessing_Pipeline.Tokenize_Vectorize import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def test_kNN(train, tra_y, test, tes_y):
        print('---------------------------------------------------------------')
        print(f'k = 1')
        print('---------------------------------------------------------------')
        kNN = KNeighborsClassifier(n_neighbors=1)
        kNN = kNN.fit(train, tra_y)
        y_pred = kNN.predict(test)
        acc = (metrics.accuracy_score(tes_y, y_pred)*100)
        print(f'> Accuracy: {acc}')
        cr = classification_report(tes_y, y_pred, output_dict=True)
        f1_score = cr['weighted avg']['f1-score']
        print(f'> F1-Score: {f1_score}')
        cm = confusion_matrix(tes_y.argmax(axis=1), y_pred.argmax(axis=1))
        cm = (cm / cm.astype(np.float).sum(axis=1))
        print(f'> Confusion Matrix:\n{cm.round(3)}')





print('Word TFIDF')
test_kNN(train_tfidf, train_y, test_tfidf, test_y)
print('Word SVD')
test_kNN(train_svd, train_y, test_svd, test_y)
print('Word LDiA')
test_kNN(train_ldia, train_y, test_ldia, test_y)
print('3-gram TFIDF')
test_kNN(trigram_tfidf_train, train_y, trigram_tfidf_test, test_y)
print('3-gram SVD')
test_kNN(trigram_svd_train, train_y, trigram_svd_test, test_y)
print('3-gram LDiA')
test_kNN(trigram_ldia_test, train_y, trigram_ldia_test, test_y)

