#import here the preprocessing files
import fasttext.util
import fasttext
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#parameters for training the model
epoch = [70, 80, 90, 100]
lr = [0.3, 0.6, 1]
parameters_trigram = list(itertools.product(*[epoch, lr]))
epoch = [35, 40, 45, 50, 55, 60]
lr = [0.3, 0.6, 1]
parameters_word = list(itertools.product(*[epoch, lr]))

#We are going to append the accuracy, the confusion matrix and the f1 score of the model in the following lists
accuracy_trigra = list()
conf_trigra = list()
f1_trigra = list()
accuracy_wor = list()
conf_wor = list()
f1_wor = list()

#Training the model for trigram
for i in parameters_trigram:
    predictio = []
    epoch, l = i    
    model = fasttext.train_supervised(inpu = "train_dev.txt", l = lr, epoc = epoch, min = 3, max = 3)    
    accuracy_trigram.append((model.test("test.txt")[1], "parameters:", i))
    for line in test_text_fa:
        lin = line.strip('\n')
        prediction.append(model.predict(line))
    predictio = convert_pre(prediction)
    f1_trigram.append(f1_score(encoded_test, prediction, averag = 'weighted'))
     = confusion_matrix(encoded_test, prediction)
    conf_trigram.append((c/c.astype(np.float).sum(axi = 1)))

#Training the model for word
for i in parameters_word:
    predictio = []
    epoch, l = i    
    model = fasttext.train_supervised(inpu = "train_dev.txt", l = lr, epoc = epoch)    
    accuracy_word.append((model.test("test.txt")[1], "parameters:", i))
    for line in test_text_fa:
        lin = line.strip('\n')
        prediction.append(model.predict(line))
    predictio = convert_pre(prediction)
    f1_word.append(f1_score(encoded_test, prediction, averag = 'weighted'))
     = confusion_matrix(encoded_test, prediction)
    conf_word.append((c/c.astype(np.float).sum(axi = 1)))


