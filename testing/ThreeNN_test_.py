from Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np





def ThreeNN_test(combinations, train, train_y, test, test_y, dims):
    for i in combinations.keys():
        print(i)
        a, b, c, d = i
        model = Sequential()
        model.add(Dense(a, input_dim=dims, activation='relu'))
        model.add(Dense(b))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(c))
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
tfidf_dict = {(64, 96, 128, 3): 1, (96, 32, 96, 3): 1, (128, 96, 256, 3): 1}
svd_dict = {(96, 96, 96, 10): 1, (96, 256, 64, 14): 1, (128, 128, 96, 10): 1}
ldia_dict = {(96, 256, 64, 8): 1, (128, 64, 96, 13): 1, (128, 96, 96, 7): 1}
clean_tfidf_dict = {(128, 96, 256, 2): 1, (128, 256, 64, 2): 1, (96, 96, 96, 4): 1}
clean_svd_dict = {(128, 63, 32, 8): 1, (128, 32, 64, 6): 1, (128, 32, 64, 5): 1}
clean_ldia_dict= {(128, 64, 256, 11): 1, (128, 96, 128, 3): 1, (32, 96, 32, 9): 1}

# Word best Meurons*Epochs
word_tfidf_dict = {(128, 32, 128, 10): 1, (96, 64, 128, 3): 1, (96, 64, 128, 6): 1}
word_svd_dict = {(128, 96, 64, 7): 1, (128, 256, 128, 5): 1, (64, 256, 64, 12): 1}
word_ldia_dict = {(32, 96, 256, 1): 1, (96, 96, 256, 1): 1, (32, 96, 128, 14): 1}


print("char 3-grams")
ThreeNN_test(tfidf_dict, trigram_tfidf_train, train_y, trigram_tfidf_test, test_y, dim_train_char)
ThreeNN_test(word_tfidf_dict, trigram_tfidf_train, train_y, trigram_tfidf_test, test_y, dim_train_char)
ThreeNN_test(clean_tfidf_dict, trigram_tfidf_train, train_y, trigram_tfidf_test, test_y, dim_train_char)

print("words")
ThreeNN_test(tfidf_dict, train_tfidf, train_y, test_tfidf, test_y, dim_train_word)
ThreeNN_test(word_tfidf_dict, train_tfidf, train_y, test_tfidf, test_y, dim_train_word)

print("LDA char 3-grams")
ThreeNN_test(ldia_dict, trigram_ldia_train, train_y, trigram_ldia_test, test_y, 32)
ThreeNN_test(word_ldia_dict, trigram_ldia_train, train_y, trigram_ldia_test, test_y, 32)
ThreeNN_test(clean_ldia_dict, trigram_ldia_train, train_y, trigram_ldia_test, test_y, 32)

print("LSA char 3-grams")
ThreeNN_test(svd_dict, trigram_svd_train, train_y, trigram_svd_test, test_y, 32)
ThreeNN_test(word_svd_dict, trigram_svd_train, train_y, trigram_svd_test, test_y, 32)
ThreeNN_test(clean_svd_dict, trigram_svd_train, train_y, trigram_svd_test, test_y, 32)

print("LDA words")
ThreeNN_test(ldia_dict, train_ldia, train_y, test_ldia, test_y, 32)
ThreeNN_test(word_ldia_dict, train_ldia, train_y, test_ldia, test_y, 32)

print("LSA words")
ThreeNN_test(svd_dict, train_svd, train_y, test_svd, test_y, 32)
ThreeNN_test(word_svd_dict, train_svd, train_y, test_svd, test_y, 32)

