from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from Tokenize_Vectorize import aria_svd_x, dummy_y_x, aria_svd_y, dummy_y_y


kNN = KNeighborsClassifier(n_neighbors=3)
kNN = kNN.fit(aria_svd_x, dummy_y_x)
y_pred = kNN.predict(aria_svd_y)
print("kNN Accuracy:", metrics.accuracy_score(dummy_y_y, y_pred))
