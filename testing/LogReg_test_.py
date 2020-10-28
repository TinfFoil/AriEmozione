from Preprocessing_Pipeline.Tokenize_Vectorize import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def LogReg_test(train, train_y, test, test_y):
    LogReg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    LogReg = LogReg.fit(train, train_y)

    y_pred = LogReg.predict(test)
    acc = (metrics.accuracy_score(test_y, y_pred)*100)
    print(f'> Accuracy: {acc}')

    cr = classification_report(test_y, y_pred, output_dict=True)
    f1_score = cr['weighted avg']['f1-score']
    print(f'> F1-Score: {f1_score}')

    cm = confusion_matrix(test_y, y_pred)
    cm = (cm / cm.astype(np.float).sum(axis=1))
    print(f'> Confusion Matrix:\n{cm.round(3)}')




print('Word TFIDF')
LogReg_test(train_tfidf, encoded_train, test_tfidf, encoded_test)
print('Word SVD')
LogReg_test(train_svd, encoded_train, test_svd, encoded_test)
print('Word LDiA')
LogReg_test(train_ldia, encoded_train, test_ldia, encoded_test)
print('3-gram TFIDF')
LogReg_test(trigram_tfidf_train, encoded_train, trigram_tfidf_test, encoded_test)
print('3-gram SVD')
LogReg_test(trigram_svd_train, encoded_train, trigram_svd_test, encoded_test)
print('3-gram LDiA')
LogReg_test(trigram_ldia_train, encoded_train, trigram_ldia_test, encoded_test)
