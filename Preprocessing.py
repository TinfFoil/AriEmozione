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
    for idx, cell in df["Text"].iteritems():  # ZAP1596906_00,ZAP1596906_01 must be corrected, they have no "Emotion" evaluation
        if 'Nessuna' in df.loc[idx, 'Emotion']:  # skipping lines with no label for Emotion
            df = df.drop(idx)
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



#for idx, cell in aria_text.iteritems(): #ZAP1596906_00,ZAP1596906_01 must be corrected, they have no "Emotion" evaluation
#  if 'Nessuna' in df.loc[idx, 'Emotion']: #skipping lines with no label for Emotion
#      print('"',df.loc[idx, 'Text'],'"', 'Number of characters: {}'.format(len(df.loc[idx, 'Text'])))