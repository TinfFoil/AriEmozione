import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from collections import Counter
from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from collections import Counter


columns = ["Names", "Text", "Emotion", "Fiducia", "?"]
df = pd.read_csv("ariaset_train.tsv", sep="\t", encoding="latin-1", names=columns)
df_dev = pd.read_csv("ariaset_dev.tsv", sep="\t", encoding="latin-1", names=columns)
df_test=pd.read_csv("ariaset_test.tsv", sep="\t", encoding="latin-1", names=columns)

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

df_test = preprocess(df_test)
test_text = df_test["Text"]
test_emotion = df_test["Emotion"]

test_emotion_list = test_emotion.tolist()
encoder = LabelEncoder()
encoder.fit(test_emotion_list)
encoded_test = encoder.transform(test_emotion_list)
dummy_test = np_utils.to_categorical(encoded_test)
print(f"Classes to be predicted: {encoder.classes_}\n", f"Corresponding number: {set(encoded_test)}\n",
      f"Instances per class: {Counter(encoded_test)}")


train_text=pd.concat([aria_text,dev_text])
train_y=np.concatenate([dummy_y,dummy_dev])
train_emotion=np.concatenate([emotion,dev_emotion])
train_y_list=np.concatenate([encoded_train,encoded_dev])
test_text=test_text
test_y=dummy_test
test_emotion=test_emotion
test_y_list=encoded_test

#preprocessing dataset for fasttext training
import fasttext.util
import fasttext
from pandas import DataFrame
import it_core_news_sm
import re
nlp = it_core_news_sm.load()
def tokenizer_FASTTEXT(doc):
    tokenize = []
    new_verse=[]
    for x in doc:
        verse = nlp(x)
        new_verse = []
        for w in verse:
            regex = re.compile(r'( +|\'|\-|\,|\!|\:|\;|\?|\.|\(|\)|\«|\»)')
            if not regex.match(w.text):
                w_lower = w.text.casefold()
                new_verse.append(w_lower)
        tokenize.append(" ".join(new_verse))

    return tokenize

train_tokenized = tokenizer_FASTTEXT(train_text)
test_tokenized = tokenizer_FASTTEXT(test_text)
#prepare dataset for fasttext
train=[]
for i,j in zip(train_emotion,train_tokenized):
    t="__label__"+i+" "+j+"\n"
    train.append(t)
file_train= open("train.txt","w")
file_train.writelines(train)
test=[]
for i,j in zip(test_emotion,test_tokenized):
    t="__label__"+i+" "+j+"\n"
    test.append(t)
file_train= open("test.txt","w")
file_train.writelines(test)

#generate combinations of parameters

import itertools
epoch=[20,25,30,35,40,45,50,55,60]
lr=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
combinations = list(itertools.product(*[epoch,lr]))

#train the model
accuracy=dict()
for i in combinations:
    epoch,lr=i    
    model = fasttext.train_supervised(input="train.txt", lr=lr, epoch=epoch,minn=3,maxn=3)    
    accuracy[i]=model.test("test.txt")[1]
print(accuracy)