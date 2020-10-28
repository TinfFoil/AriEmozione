#import preprocessing file and vectorize_tokenize here
from sklearn.svm import SVC
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from Preprocessing import *
from Tokenize_Vectorize import *


#generate class weight
class_weight = class_weight.compute_class_weight("balanced", np.unique(train_y_list), train_y_list)
class_weight
a = [0, 1, 2, 3, 4, 5]
weight = dict()
for k, v in zip(a, class_weight):
    weight[k] = v

#this module trains the model using train set and development set and then tests the model using the test set
def svc_loop(train_x, test_x, para):
    acc = []
    acc_std = []
    f1 = []
    f1_std = []
    cm = []
    re_acc = []
    re_f1 = []
    re_cm = []
    for i in para:
        print("Test combination", i)
        c, gamma = i
        try_svc = SVC(kernel = 'rbf', C = c, class_weight = weight, gamma = gamma)
        y_pred = try_svc.fit(train_x, train_y_list).predict(test_x)
        f1.append(f1_score(encoded_test, y_pred, average = 'weighted'))
        acc.append(try_svc.score(test_x, encoded_test))
        zzz = confusion_matrix(encoded_test, y_pred)
        cm.append(zzz/zzz.astype(np.float).sum(axis = 1))
    return f1, acc, cm

c = [1000, 100, 10, 1]
gamma = [0.001, 0.01]
parameters = list(itertools.product(*[c, gamma]))

f1_tfidf, acc_tfidf, cm_tfidf = svc_loop(train_tfidf, test_tfidf, parameters)
f1_svd, acc_svd, cm_svd = svc_loop(train_svd, test_svd, parameters)
f1_ldia, acc_ldia, cm_ldia = svc_loop(train_ldia, test_ldia, parameters)
f1_tfidf_trigram, acc_tfidf_trigram, cm_tfidf_trigram = svc_loop(trigram_tfidf_train, trigram_tfidf_test, parameters)
f1_svd_trigram, acc_svd_trigram, cm_svd_trigram = svc_loop(trigram_svd_train, trigram_svd_test, parameters)
f1_ldia_trigram, acc_ldia_trigram, cm_ldia_trigram = svc_loop(trigram_ldia_train, trigram_ldia_test, parameters)

for i, f1, acc, conf in zip(parameters, f1_tfidf, acc_tfidf, cm_tfidf):
    c,g=i
    print('WORD TFIDF Test Result--------------------------')
    print(f'> With C: {c}, Gamma:{g}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')

for i, f1, acc, conf in zip(parameters, f1_svd, acc_svd, cm_svd):
    c,g=i
    print('WORD SVD Test Result--------------------------')
    print(f'> With C: {c}, Gamma:{g}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
    
for i, f1, acc, conf in zip(parameters,f1_ldia, acc_ldia, cm_ldia):
    c,g=i
    print('WORD LDIA Test Result--------------------------')
    print(f'> With C: {c}, Gamma:{g}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')

for i, f1, acc, conf in zip(parameters,f1_tfidf_trigram, acc_tfidf_trigram, cm_tfidf_trigram):
    c,g=i
    print('TRIGRAM TFIDF Test Result--------------------------')
    print(f'> With C: {c}, Gamma:{g}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
    
for i, f1, acc, conf in zip(parameters,f1_svd_trigram, acc_svd_trigram, cm_svd_trigram):
    c,g=i
    print('TRIGRAM SVD Test Result--------------------------')
    print(f'> With C: {c}, Gamma:{g}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
    
for i, f1, acc, conf in zip(parameters,f1_ldia_trigram, acc_ldia_trigram, cm_ldia_trigram):
    c,g=i
    print('TRIGRAM LDIA Test Result--------------------------')
    print(f'> With C: {c}, Gamma:{g}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
