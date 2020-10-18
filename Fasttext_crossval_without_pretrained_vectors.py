from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import fasttext.util
import fasttext
import itertools
import statistics
from Preprocessing import *
from Tokenize_Vectorize import *
from Fasttext_preprocessing import *

cv_tokenized = tokenizer_FASTTEXT(cv_text)
dev_tokenized = tokenizer_FASTTEXT(dev_text)

#fasttext classifier using trigram
def fasttext_ngram(para, n):
    acc = list()
    cm = list()
    f1_scores = list()
    acc_std = list()
    f1_std = list()
    dataset = np.concatenate((cv_tokenized, dev_tokenized), axis = 0)
    label = np.concatenate((emotion, dev_emotion), axis = 0)
    kf = KFold(n_splits = 10)
    re_acc = list()
    re_f1 = list()
    re_cm = list()
    for i in para:        
        epoch, lr = i
        n = n
        for train, test in kf.split(dataset):
            predictions = []
            test_label_num = convert_emotion_list_to_string_of_numbers(label[test])
            fn_train = label_data(label[train], dataset[train], "train")
            fn_test = label_data(label[test], dataset[test], "test")
            Y = label_data_return_list(label[test], dataset[test])
            model = fasttext.train_supervised(input = fn_train, lr = lr, epoch = epoch, minn = n, maxn = n)
            for line in Y:
                line = line.strip('\n')
                predictions.append(model.predict(line))
            pred = convert_pre(predictions)
            f1_scores.append(f1_score(test_label_num, pred, average = 'weighted'))
            conf = confusion_matrix(test_label_num, pred)
            cm.append(conf/conf.astype(np.float).sum(axis = 1))
            acc.append(model.test(fn_test)[1])
        ave_acc = Average(acc)
        ave_f1 = Average(f1_scores)
        ave_cm = np.nanmean(cm, axis = 0)
        re_acc.append(ave_acc)
        re_f1.append(ave_f1)
        re_cm.append(ave_cm)
        acc_std.append(statistics.stdev(acc))
        f1_std.append(statistics.stdev(f1_scores))
    return re_acc, re_f1, re_cm, acc_std, f1_std

#fasttext classifier using word
def fasttext_word(para):
    acc = list()
    cm = list()
    f1_scores = list()       
    dataset = np.concatenate((cv_tokenized, dev_tokenized), axis = 0)
    label = np.concatenate((emotion, dev_emotion), axis = 0)
    kf = KFold(n_splits = 10)
    re_acc = list()
    re_f1 = list()
    re_cm = list()
    acc_std = list()
    f1_std = list()
    for i in para:        
        epoch, lr = i
        for train, test in kf.split(dataset):
            predictions = []
            test_label_num = convert_emotion_list_to_string_of_numbers(label[test])
            fn_train = label_data(label[train], dataset[train], "train")
            fn_test = label_data(label[test], dataset[test], "test")
            Y = label_data_return_list(label[test], dataset[test])
            mode = fasttext.train_supervised(input = fn_train, lr = lr, epoch = epoch)
            for line in Y:
                line = line.strip('\n')
                predictions.append(model.predict(line))
            pred = convert_pre(predictions)
            f1_scores.append(f1_score(test_label_num, pred, average = 'weighted'))
            conf = confusion_matrix(test_label_num, pred)
            cm.append(conf/conf.astype(np.float).sum(axis = 1))
            acc.append(model.test(fn_test)[1])
        ave_acc = Average(acc)
        ave_f1 = Average(f1_scores)
        ave_cm = np.nanmean(cm, axis = 0)
        re_acc.append(ave_acc)
        re_f1.append(ave_f1)
        re_cm.append(ave_cm)
        acc_std.append(statistics.stdev(acc))
        f1_std.append(statistics.stdev(f1_scores))
    return re_acc, re_f1, re_cm, acc_std, f1_std

#generate the parameters for training the model
epoch = [70, 80, 90, 100]
lr = [0.3, 0.6, 1]
parameters_trigram = list(itertools.product(*[epoch, lr]))
epoch = [35, 40, 45, 50, 55, 60]
lr = [0.3, 0.6, 1]
parameters_word = list(itertools.product(*[epoch, lr]))

#train the model
accuracy_trigram, f1_trigram, conf_trigram, acc_std_trigram, f1_std_trigram = fasttext_ngram(parameters_trigram, 3)
accuracy_word, f1_word, conf_word, acc_std_word, f1_std_word = fasttext_word(parameters_word)
for i, acc, f1, conf, accstd, f1std in zip(parameters_trigram,accuracy_trigram, f1_trigram, conf_trigram, acc_std_trigram, f1_std_trigram):
    e,l=i
    print('Trigram Test Result--------------------------')
    print(f'> With Epoch: {e},Learning rate:{l}')
    print(f'> Accuracy: {acc})')
    print(f'> Standard deviation of accuracy: {accstd})')
    print(f'> F1-Score: {f1}')
    print(f'> Standard deviation of f1: {f1std})')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')

for i, acc, f1, conf, accstd, f1std in zip(parameters_word, accuracy_word, f1_word, conf_word, acc_std_word, f1_std_word):
    e,l=i
    print('Word Test Result--------------------------')
    print(f'> With Epoch: {e},Learning rate:{l}')
    print(f'> Accuracy: {acc})')
    print(f'> Standard deviation of accuracy: {accstd})')
    print(f'> F1-Score: {f1}')
    print(f'> Standard deviation of f1: {f1std})')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')