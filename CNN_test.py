import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from collections import Counter

columns = ["Names", "Text", "Emotion", "Fiducia", "?"]
df = pd.read_csv("ariaset_train.tsv", sep="\t", encoding="utf-8", names=columns)
df_dev = pd.read_csv("ariaset_dev.tsv", sep="\t", encoding="utf-8", names=columns)


def preprocess(data):
    df = data.drop(['?'], axis=1)
    df = df.dropna()
    df = df.sample(frac=1)
    return df


df = preprocess(df)
aria_text = df["Text"]
emotion = df["Emotion"]

emotion_list = emotion.tolist()
encoder = LabelEncoder()
encoder.fit(emotion_list)
encoded_train = encoder.transform(emotion_list)
dummy_y = np_utils.to_categorical(encoded_train)
print(f"Classes to be predicted: {encoder.classes_}\n", f"Corresponding number: {set(encoded_train)}\n",
      f"Instances per class: {Counter(encoded_train)}")

df_dev = preprocess(df_dev)
dev_text = df_dev["Text"]
dev_emotion = df_dev["Emotion"]

dev_emotion_list = dev_emotion.tolist()
encoder = LabelEncoder()
encoder.fit(dev_emotion_list)
encoded_dev = encoder.transform(dev_emotion_list)
dummy_dev = np_utils.to_categorical(encoded_dev)
print(f"Classes to be predicted: {encoder.classes_}\n", f"Corresponding number: {set(encoded_dev)}\n",
      f"Instances per class: {Counter(encoded_dev)}")

# create the TF-IDF + SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import it_core_news_sm

nlp = it_core_news_sm.load()
tokenized = []
count = 0
for verse in aria_text:
    doc = nlp(verse)
    tokenized.append([])
    for w in doc:
        tokenized[count].append(w.text)
    count += 1
print(tokenized)  # here I was checking that the tokenization was alright, there's just a little bit of noise
# due to spaces and punctuation

total = [len(x) for x in tokenized]
total = sorted(total)
print(total[-5:])
y = 0
for x in total:
    y += x
avg_len = y / len(total)  # 18
max_len = 20  # to even it out nicely


def italian_tokenizer(verse):
    tokenized = []
    doc = nlp(verse)   #we could add here .casefold() to make it case insensitive
    for w in doc:
        tokenized.append(w.text)
    return tokenized


vectorizer = TfidfVectorizer(tokenizer=italian_tokenizer)  # Now it seems to be working
aria_tfidf = vectorizer.fit_transform(aria_text).toarray()
dev_tfidf = vectorizer.transform(dev_text).toarray()
aria_tfidf = pd.DataFrame(aria_tfidf)
dev_tfidf = pd.DataFrame(dev_tfidf)
print(vectorizer.get_feature_names(), f"\nLenght = {len(vectorizer.get_feature_names())}")
print(vectorizer.vocabulary_, f"\nLenght = {len(vectorizer.vocabulary_)}")

print("This is the Train tfidf:\n", aria_tfidf)
print("This is the Dev tfidf:\n", dev_tfidf)


sc = StandardScaler()
#create the topic vectors using truncated SVD
svd = TruncatedSVD(n_components=32, n_iter=100)
aria_svd = svd.fit_transform(aria_tfidf)
aria_svd = pd.DataFrame(aria_svd)
aria_svd = sc.fit_transform(aria_svd)
print("This is the Train SVD:\n", aria_svd, f"\nShape: {aria_svd.shape}")


dev_svd = svd.transform(dev_tfidf)
dev_svd = pd.DataFrame(dev_svd)
dev_svd = sc.fit_transform(dev_svd)
print("This is the Dev SVD:\n", dev_svd, f"\nShape: {dev_svd.shape}")

# Padding/truncating the data (if necessary)
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split

aria_svd_x, aria_svd_y, dummy_y_x, dummy_y_y = train_test_split(aria_svd, dummy_y,
                                                                test_size=0.2)  # here I am just splitting the training dataset


# to mimic

# Method to pad or truncate the input
def pad_trunc(data, maxlen):
    """
    For a given dataset pad with zero vectors or truncate to maxlen
    """
    new_data = []
    # Create a vector of 0s the length of our word vectors
    zero_vector = []
    for _ in range(len(data)):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                np.append(temp, zero_vector)
        else:
            temp = sample
        new_data.append(temp)

    return new_data

embedding_dims = 32
filters = 250  # (!)
kernel_size = 3
hidden_dims = 250


x_train = pad_trunc(aria_svd_x, max_len)
x_test = pad_trunc(aria_svd_y, max_len)

x_train = np.reshape(x_train, (len(x_train), max_len, embedding_dims))
y_train = np.array(dummy_y_x)
x_test = np.reshape(x_test, (len(x_test), max_len, embedding_dims))
y_test = np.array(dummy_y_y)




print('Building model...')
model = Sequential()  # The standard NN model
model.add(Conv1D(  # Adding a convolutional layer
    filters,
    kernel_size,
    padding='valid',  # in this example the output is going to be lightly smaller
    activation='relu',
    strides=1,  # the shift
    input_shape=(max_len, embedding_dims))
)
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(7, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(aria_svd_x, dummy_y_x, nb_epoch=10, batch_size=32, verbose=1)

score = model.evaluate(aria_svd_x, dummy_y_y, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
