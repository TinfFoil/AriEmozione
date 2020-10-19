import fasttext.util
import fasttext
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from Tokenize_Vectorize import *
from Fasttext_preprocessing import *
import itertools

#parameters for training the model
epoch = [70, 80, 90, 100]
lr = [0.3, 0.6, 1]
parameters_trigram = list(itertools.product(*[epoch, lr]))
epoch = [35, 40, 45, 50, 55, 60]
lr = [0.3, 0.6, 1]
parameters_word = list(itertools.product(*[epoch, lr]))

#We are going ton append the accuracy, the confusion matrix and the f1 score of the model in the following lists
accuracy_trigram = list()
conf_trigram = list()
f1_trigram = list()
accuracy_word = list()
conf_word = list()
f1_word = list()

#Training the model for trigram
for i in parameters_trigram:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train.txt", lr = lr, epoch = epoch, minn = 3, maxn= 3)    
    accuracy_trigram.append((model.test("test.txt")[1], "parameters:", i))
    for line in test_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_trigram.append(f1_score(encoded_test, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_test, prediction)
    conf_trigram.append((c/c.astype(np.float).sum(axi = 1)))

#Training the model for word
for i in parameters_word:
    prediction = []
    epoch, lr = i    
    model = fasttext.train_supervised(input = "train.txt", lr = lr, epoch = epoch)    
    accuracy_word.append((model.test("test.txt")[1], "parameters:", i))
    for line in test_text_fa:
        line = line.strip('\n')
        prediction.append(model.predict(line))
    prediction = convert_pre(prediction)
    f1_word.append(f1_score(encoded_test, prediction, average = 'weighted'))
    c = confusion_matrix(encoded_test, prediction)
    conf_word.append((c/c.astype(np.float).sum(axi = 1)))
    
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
