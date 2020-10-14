#import preprocessing file and vetorizer_tokeniz file
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#generate class weight
class_weight = class_weight.compute_class_weight("balanced", np.unique(encoded_train), encoded_train)
class_weight
 = [0, 1, 2, 3, 4, 5]
weigh = dict()
for k, v in zip(a, class_weight):
    weight[k = v

y_lis = np.concatenate([encoded_train, encoded_dev])

#the following module is used to cross validate the train dataset and the development dataset. It returns the f1 score, the accuracy and the confusion matrix
def svc_loop(train_x, test_x, para):
    acc_st = [] #the standard deviation of accuracy
    f1_st = [] #the standard deviation of f1 score
    re_ac = []
    re_f = []
    re_c = []
     = np.concatenate((train_x, test_x), axi = 0)
     = np.concatenate((encoded_train, encoded_dev), axi = 0)
    for i in para:
        ac = []
        f = []
        c = []
        kf = KFold(n_split = 10)
        c, gamm = i
        for train, test in kf.split(X, Y):
            try_svc = SVC(kerne = 'rbf', = c, class_weigh = weight, gamm = gamma)
            y_pred = try_svc.fit(X[train], Y[train]).predict(X[test])
            f1.append(f1_score(Y[test], y_pred, averag = 'weighted'))
            acc.append(try_svc.score(X[test], Y[test]))
             = confusion_matrix(Y[test], y_pred)
            cm.append(i/i.astype(np.float).sum(axi = 1))
        re_f1.append(np.mean(f1))
        re_acc.append(np.mean(acc))
        re_cm.append(np.nanmean(cm, axi = 0))
        acc_std.append(np.std(acc))
        f1_std.append(np.std(f1))
    return re_f1, re_acc, re_cm, acc_std, f1_std

c = [1000, 100, 10, 1]
gamma = [0.001, 0.01]
parameters = list(itertools.product(*[c, gamma]))

f1_tfidf, acc_tfidf, cm_tfidf, acc_std_tfidf, f1_std_tfidf = svc_loop(aria_tfidf, dev_tfidf, parameters)
f1_svd, acc_svd, cm_svd, acc_std_svd, f1_std_svd = svc_loop(aria_svd, dev_svd, parameters)
f1_ldia, acc_ldia, cm_ldia, acc_std_ldia, f1_std_ldia = svc_loop(aria_ldia, dev_ldia, parameters)
f1_tfidf_trigram, acc_tfidf_trigram, cm_tfidf_trigram, acc_std_tfidf_trigram, f1_std_tfidf_trigra = svc_loop(trigram_tfidf, trigram_tfidf_dev, parameters)
f1_svd_trigram, acc_svd_trigram, cm_svd_trigram, acc_std_svd_trigram, f1_std_svd_trigra = svc_loop(trigram_svd, trigram_svd_dev, parameters)
f1_ldia_trigram, acc_ldia_trigram, cm_ldia_trigram, acc_std_ldia_trigram, f1_std_ldia_trigra = svc_loop(trigram_ldia, trigram_ldia_dev, parameters)