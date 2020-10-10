#import fasttext preprocessing file and other files here
import fasttext.util
import fasttext
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#fasttext with pre-trained character trigram vectors. Trained using train set and tested using dev set
accuracy_trigram = list()
conf_trigram = list()
f1_trigram = list()
parameters_trigram = [(1, 0.3), 
 (1, 0.6), 
 (1, 1), 
 (3, 0.3), 
 (3, 0.6), 
 (3, 1), 
 (5, 0.3)]
for i in parameters_trigram:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train.txt", lr = lr, epoch = epoch, minn = 3, maxn = 3, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec")
    accuracy_trigram.append((model.test("dev.txt")[1], "parameters:", i))
    for line in dev_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_trigram.append(f1_score(encoded_dev, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_dev, prediction)
    conf_trigram.append((c/c.astype(np.float).sum(axis = 1)))

    
#fasttext with pre-trained word vectors. Trained using train set and tested using dev set
accuracy_word = list()
conf_word = list()
f1_word = list()
parameters_word = [(1, 0.3), 
 (1, 0.6), 
 (1, 1), 
 (3, 0.3), 
 (3, 0.6), 
 (3, 1), 
 (5, 0.3), 
 (5, 0.6), 
 (5, 1), (6, 0.3), (7, 0.3)]
for i in parameters_word:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train.txt", lr = lr, epoch = epoch, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec") 
    accuracy_word.append((model.test("dev.txt")[1], "parameters:", i))
    for line in dev_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_word.append(f1_score(encoded_dev, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_dev, prediction)
    conf_word.append((c/c.astype(np.float).sum(axis = 1)))

#fasttext with pre-trained character trigram vectors. Trained using train set and dev set, tested using test set
accuracy_trigram_test = list()
conf_trigram_test = list()
f1_trigram_test = list()
parameters_trigram_test = [(3, 0.6)]
for i in parameters_trigram_test:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train_dev.txt", lr = lr, epoch = epoch, minn = 3, maxn = 3, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec")    
    accuracy_trigram_test.append((model.test("test.txt")[1]))
    for line in test_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_trigram_test.append(f1_score(encoded_test, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_test, prediction)
    conf_trigram_test.append((c/c.astype(np.float).sum(axis = 1)))

    
#fasttext with pre-trained word vectors. Trained using train set and dev set, tested using test set
accuracy_word_test = list()
conf_word_test = list()
f1_word_test = list()
parameters_word_test = [(5, 0.6)]
for i in parameters_word_test:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train_dev.txt", lr = lr, epoch = epoch, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec")    
    accuracy_word_test.append((model.test("test.txt")[1]))
    for line in test_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_word_test.append(f1_score(encoded_test, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_test, prediction)
    conf_word_test.append((c/c.astype(np.float).sum(axis = 1)))
