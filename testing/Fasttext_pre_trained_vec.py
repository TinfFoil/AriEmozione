import fasttext.util
import fasttext
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import itertools
from Preprocessing_Pipeline.Tokenize_Vectorize import *
from Preprocessing_Pipeline.Fasttext_preprocessing import *

#fasttext with pre-trained character trigram vectors. Trained using cv set and tested using dev set
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
    model = fasttext.train_supervised(input = "cv.txt", lr = lr, epoch = epoch, minn = 3, maxn = 3, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec")
    accuracy_trigram.append((model.test("dev.txt")[1]))
    for line in dev_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_trigram.append(f1_score(encoded_dev, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_dev, prediction)
    conf_trigram.append((c/c.astype(np.float).sum(axis = 1)))
    

    
#fasttext with pre-trained word vectors. Trained using cv set and tested using dev set
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
    model = fasttext.train_supervised(input = "cv.txt", lr = lr, epoch = epoch, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec") 
    accuracy_word.append((model.test("dev.txt")[1]))
    for line in dev_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_word.append(f1_score(encoded_dev, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_dev, prediction)
    conf_word.append((c/c.astype(np.float).sum(axis = 1)))

#fasttext with pre-trained character trigram vectors. Trained using cv set and dev set, tested using test set
accuracy_trigram_test = list()
conf_trigram_test = list()
f1_trigram_test = list()
parameters_trigram_test = [(3, 0.6)]
for i in parameters_trigram_test:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train.txt", lr = lr, epoch = epoch, minn = 3, maxn = 3, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec")    
    accuracy_trigram_test.append((model.test("test.txt")[1]))
    for line in test_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_trigram_test.append(f1_score(encoded_test, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_test, prediction)
    conf_trigram_test.append((c/c.astype(np.float).sum(axis = 1)))

    
#fasttext with pre-trained word vectors. Trained using cv set and dev set, tested using test set
accuracy_word_test = list()
conf_word_test = list()
f1_word_test = list()
parameters_word_test = [(5, 0.6)]
for i in parameters_word_test:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train.txt", lr = lr, epoch = epoch, dim = 300, pretrainedVectors = "D:/vec/cc.it.300.vec/cc.it.300.vec")    
    accuracy_word_test.append((model.test("test.txt")[1]))
    for line in test_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_word_test.append(f1_score(encoded_test, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_test, prediction)
    conf_word_test.append((c/c.astype(np.float).sum(axis = 1)))
    
for i, acc, f1, conf in zip(parameters_trigram, accuracy_trigram, f1_trigram, conf_trigram):
    e,l=i
    print('Trigram Test Result--------------------------')
    print(f'> With Epoch: {e},Learning rate:{l}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
    
for i, acc, f1, conf in zip(parameters_word, accuracy_word, f1_word, conf_word):
    e,l=i
    print('Word Test Result--------------------------')
    print(f'> With Epoch: {e},Learning rate:{l}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
    
for i, acc, f1, conf in zip(parameters_trigram_test, accuracy_trigram_test, f1_trigram_test, conf_trigram_test):
    e,l=i
    print('Trigram Test Result--------------------------')
    print(f'> With Epoch: {e},Learning rate:{l}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
    
for i, acc, f1, conf in zip(parameters_word_test, accuracy_word_test, f1_word_test, conf_word_test):
    e,l=i
    print('Word Test Result--------------------------')
    print(f'> With Epoch: {e},Learning rate:{l}')
    print(f'> Accuracy: {acc})')
    print(f'> F1-Score: {f1}')
    print(f'> Confusion Matrix:\n{conf}')
    print('------------------------------------------------------------------------')
