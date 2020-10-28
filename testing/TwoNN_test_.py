from Preprocessing_Pipeline.Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np






def TwoNN_test(combinations, train, train_y, test, test_y, dims):
    for i in combinations.keys():
        print(i)
        a, b, d = i
        model = Sequential()
        model.add(Dense(a, input_dim=dims, activation='relu'))
        model.add(Dense(b))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train, train_y, epochs=d, batch_size=32, verbose=0)
        print(f'-----------------------------------------------------------------------')
        # Generate generalization metrics
        scores = model.evaluate(test, test_y, verbose=0)
        print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

        Y = np.argmax(test_y, axis=1)  # here we build the precision/recall/f1 table for each fold
        y_pred = model.predict_classes(test)
        cr = classification_report(Y, y_pred, output_dict=True)
        cm = confusion_matrix(Y, y_pred)
        f1_score = cr['weighted avg']['f1-score']
        print(f'> F1-Score: {f1_score}')
        cm = cm / cm.astype(np.float).sum(axis=1)
        print(f'> Confusion Matrix:\n{cm.round(3)}')
        print(f'-----------------------------------------------------------------------')


# 3-gram best Neurons*Epochs
tfidf_dict = {(32, 96, 4): 1, (16, 64, 6): 1, (64, 16, 5): 1}

print("char 3-grams")
TwoNN_test(tfidf_dict, trigram_tfidf_train, train_y, trigram_tfidf_test, test_y, dim_train_char)

print("words")
TwoNN_test(tfidf_dict, train_tfidf, train_y, test_tfidf, test_y, dim_train_word)

print("LSA char 3-grams")
TwoNN_test(tfidf_dict, trigram_svd_train, train_y, trigram_svd_test, test_y, 32)
