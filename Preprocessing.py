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


df_dev = preprocess(df_dev)
dev_text = df_dev["Text"]
dev_emotion = df_dev["Emotion"]

dev_emotion_list = dev_emotion.tolist()
encoder = LabelEncoder()
encoder.fit(dev_emotion_list)
encoded_dev = encoder.transform(dev_emotion_list)
dummy_dev = np_utils.to_categorical(encoded_dev)


