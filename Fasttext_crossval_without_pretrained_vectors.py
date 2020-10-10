#import here the preprocessing file

import fasttext.util
import fasttext
import itertools
#
train_tokenized = tokenizer_FASTTEXT(aria_text)
dev_tokenized = tokenizer_FASTTEXT(dev_text)

#fasttext classifier using trigram
def fasttext_ngram(para, n):
    ac = list()
    c = list()
    f1_score = list()
    acc_st = list()
    f1_st = list()
    datase = np.concatenate((train_tokenized, dev_tokenized), axi = 0)
    labe = np.concatenate((emotion, dev_emotion), axi = 0)
    kf = KFold(n_split = 10)
    re_ac = list()
    re_f = list()
    re_c = list()
    for i in para:        
        epoch, l = i
         = n
        for train, test in kf.split(dataset):
            prediction = []
            test_label_nu = convert_emotion_list_to_string_of_numbers(label[test])
            fn_trai = label_data(label[train], dataset[train], "train")
            fn_tes = label_data(label[test], dataset[test], "test")
             = label_data_return_list(label[test], dataset[test])
            mode = fasttext.train_supervised(inpu = fn_train, l = lr, epoc = epoch, min = n, max = n)
            for line in Y:
                lin = line.strip('\n')
                predictions.append(model.predict(line))
            pre = convert_pre(predictions)
            f1_scores.append(f1_score(test_label_num, pred, averag = 'weighted'))
            con = confusion_matrix(test_label_num, pred)
            cm.append(conf/conf.astype(np.float).sum(axi = 1))
            acc.append(model.test(fn_test)[1])
        ave_ac = Average(acc)
        ave_f = Average(f1_scores)
        ave_c = np.nanmean(cm, axi = 0)
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
    dataset = np.concatenate((train_tokenized, dev_tokenized), axi = 0)
    label = np.concatenate((emotion, dev_emotion), axi = 0)
    kf = KFold(n_split = 10)
    re_acc = list()
    re_f1 = list()
    re_cm = list()
    acc_std = list()
    f1_std = list()
    for i in para:        
        epoch, l = i
        for train, test in kf.split(dataset):
            prediction = []
            test_label_nu = convert_emotion_list_to_string_of_numbers(label[test])
            fn_trai = label_data(label[train], dataset[train], "train")
            fn_tes = label_data(label[test], dataset[test], "test")
             = label_data_return_list(label[test], dataset[test])
            mode = fasttext.train_supervised(inpu = fn_train, l = lr, epoc = epoch)
            for line in Y:
                lin = line.strip('\n')
                predictions.append(model.predict(line))
            pre = convert_pre(predictions)
            f1_scores.append(f1_score(test_label_num, pred, averag = 'weighted'))
            con = confusion_matrix(test_label_num, pred)
            cm.append(conf/conf.astype(np.float).sum(axi = 1))
            acc.append(model.test(fn_test)[1])
        ave_ac = Average(acc)
        ave_f = Average(f1_scores)
        ave_c = np.nanmean(cm, axi = 0)
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
