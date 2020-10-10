#import preprocessing file and vectorize_tokenize here
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils import class_weight


#generate class weight
class_weight = class_weight.compute_class_weight("balanced", np.unique(train_y_list), train_y_list)
class_weight
 = [0, 1, 2, 3, 4, 5]
weigh = dict()
for k, v in zip(a, class_weight):
    weight[k = v
weight

#this module trains the model using train set and development set and then tests the model using the test set
def svc_loop(train_x, test_x, para):
    ac = []
    acc_st = []
    f = []
    f1_st = []
    c = []
    re_ac = []
    re_f = []
    re_c = []
    for i in para:
        print("Test combination", i)
        c, gamm = i
        try_svc = SVC(kerne = 'rbf', = c, class_weigh = weight, gamm = gamma)
        y_pred = try_svc.fit(train_x, train_y_list).predict(test_x)
        f1.append(f1_score(test_y_list, y_pred, averag = 'weighted'))
        acc.append(try_svc.score(test_x, test_y_list))
         = confusion_matrix(test_y_list, y_pred)
        cm.append(a/a.astype(np.float).sum(axi = 1))
    return f1, acc, cm

c = [1000, 100, 10, 1]
gamma = [0.001, 0.01]
parameters = list(itertools.product(*[c, gamma]))

f1_tfidf, acc_tfidf, cm_tfidf = svc_loop(train_tfidf, test_tfidf, parameters)
f1_svd, acc_svd, cm_svd = svc_loop(train_svd, test_svd, parameters)
f1_ldia, acc_ldia, cm_ldia = svc_loop(train_ldia, test_ldia, parameters)
f1_tfidf_trigram, acc_tfidf_trigram, cm_tfidf_trigram = svc_loop(trigram_tfidf, trigram_tfidf_test, parameters)
f1_svd_trigram, acc_svd_trigram, cm_svd_trigram = svc_loop(trigram_svd, trigram_svd_test, parameters)
f1_ldia_trigram, acc_ldia_trigram, cm_ldia_trigram = svc_loop(trigram_ldia, trigram_ldia_test, parameters)
