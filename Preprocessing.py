import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from functools import reduce
from keras.utils import np_utils

columns = ["Names", "Text", "Emotion", "Fiducia", "?"]

df_cv = pd.read_csv("ariaset_train.tsv", sep="\t", encoding="latin-1", names=columns)
df_dev = pd.read_csv("ariaset_dev.tsv", sep="\t", encoding="latin-1", names=columns)
df_test = pd.read_csv("ariaset_test.tsv", sep="\t", encoding="latin-1", names=columns)


# Simple function to correctly set up the df
def preprocess(data):
    df = data.drop(['?'], axis=1)
    df = df.dropna()
    df = df.sample(frac=1)
    for idx, cell in df[
        "Text"].iteritems():
        if 'Nessuna' in df.loc[idx, 'Emotion']:  # skipping over lines with no label for Emotion
            df = df.drop(idx)
    return df


# create labels for the crossval train dataset
df_cv = preprocess(df_cv)
cv_text = df_cv["Text"]
emotion = df_cv["Emotion"]

emotion_list = emotion.tolist()
encoder = LabelEncoder()
encoder.fit(emotion_list)

encoded_cv = encoder.transform(emotion_list)

cv_y = np_utils.to_categorical(encoded_cv)

# create labels for the dev dataset
df_dev = preprocess(df_dev)
dev_text = df_dev["Text"]
dev_emotion = df_dev["Emotion"]

dev_emotion_list = dev_emotion.tolist()
encoder = LabelEncoder()
encoder.fit(dev_emotion_list)
encoded_dev = encoder.transform(dev_emotion_list)

dev_y = np_utils.to_categorical(encoded_dev)

# create labels for the test dataset
df_test = preprocess(df_test)
test_text = df_test["Text"]
test_emotion = df_test["Emotion"]

test_emotion_list = test_emotion.tolist()
encoder = LabelEncoder()
encoder.fit(test_emotion_list)
encoded_test = encoder.transform(test_emotion_list)

test_y = np_utils.to_categorical(encoded_test)

# create labels for the final train dataset
df_train = pd.concat([cv_text, dev_text])
train_y = np.concatenate([cv_y, dev_y])
train_y_list = np.concatenate([encoded_cv, encoded_dev])



# Following is a series of helper functions to be used for Fasttext

# This module returns a list of verses labelled in fasttext format
def label_data_return_list(results, data):
    labelled = list()
    for i, j in zip(results, data):
        t = "__label__" + i + " " + j + "\n"
        labelled.append(t)
    return labelled


# Returns a list of numbers representing the emotions
def convert_pre(pre):
    y_pred = list()
    for r in pre:
        if r[0][0] == '__label__Ammirazione':
            y_pred.append(0)
        if r[0][0] == '__label__Amore':
            y_pred.append(1)
        if r[0][0] == '__label__Gioia':
            y_pred.append(2)
        if r[0][0] == '__label__Paura':
            y_pred.append(3)
        if r[0][0] == '__label__Rabbia':
            y_pred.append(4)
        if r[0][0] == '__label__Tristezza':
            y_pred.append(5)
    return y_pred

# This function attaches the correct labels
def label_data(results, data, filename):
    labelled = list()
    for i, j in zip(results, data):
        t = "__label__" + i + " " + j + "\n"
        labelled.append(t)
    fn = filename + ".txt"
    fi = open(fn, "w")
    fi.writelines(labelled)
    return fn


def convert_emotion_list_to_string_of_numbers(emotion):
    emotion_list = emotion.tolist()
    encoder = LabelEncoder()
    encoder.fit(emotion_list)
    encoded = encoder.transform(emotion_list)
    return encoded


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)
