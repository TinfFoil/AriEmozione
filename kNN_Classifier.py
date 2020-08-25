from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Tokenize_Vectorize import *
from sklearn.metrics import classification_report, confusion_matrix
import pickle

best_val = list()
for n in range(1,10):
    kNN = KNeighborsClassifier(n_neighbors=n)
    kNN = kNN.fit(aria_ldia, dummy_y)
    y_pred = kNN.predict(dev_ldia)
    acc = metrics.accuracy_score(dummy_dev, y_pred)
    print(f"kNN Accuracy with k={n}:", acc)

    print(f"kNN per Class Report with k={n}:\n", classification_report(dummy_dev, y_pred))

    cm = confusion_matrix(dummy_dev.argmax(axis=1), y_pred.argmax(axis=1))
    cm = cm / cm.astype(np.float).sum(axis=1)
    print(f"kNN Confusion Matrix with k={n}:\n", cm.round(2))

    if len(best_val) == 0:
        knnPickle = open(f'Word_kNN_ldia_{n}_neighbors', 'wb')
        pickle.dump(kNN, knnPickle)
        best_val.append(acc)

    elif acc > best_val[0]:
        del best_val[0]
        knnPickle = open(f'Word_kNN_ldia_{n}_neighbors', 'wb')
        pickle.dump(kNN, knnPickle)

    else:
        pass