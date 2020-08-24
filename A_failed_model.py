#!/usr/bin/env python
# coding: utf-8

# In[209]:


import numpy as np
import pandas as pd
columns=["Names","Text","Emotion","Fiducia","?"]
df = pd.read_csv("ariaset_train.tsv",sep="\t",encoding="utf-8",names=columns)
df=df.drop(['?'], axis=1)
#delete lines that have no emotion predicted
df=df.dropna()
df=df.sample(frac=1)
aria_text=df["Text"]
emotion=df["Emotion"]


# In[151]:


df


# In[154]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
vectorizer = TfidfVectorizer(stop_words={"italian"})
aria_tfidf = vectorizer.fit_transform(aria_text)


# In[174]:


#create the TF-IDF vector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
vectorizer = TfidfVectorizer()
aria_tfidf = vectorizer.fit_transform(aria_text)
print(aria_tfidf.shape)
#create the topic vectors using truncated SVD
svd=TruncatedSVD(n_components=16,n_iter=100)
aria_svd=svd.fit_transform(aria_tfidf)
aria_svd=pd.DataFrame(aria_svd)
aria_svd.round(3)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
aria_svd=sc.fit_transform(aria_svd)


# In[175]:


#creat one hot vector of expected output
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from collections import Counter
emotion_list=emotion.tolist()
encoder=LabelEncoder()
encoder.fit(emotion_list)
encoded_Y=encoder.transform(emotion_list)
dummy_y=np_utils.to_categorical(encoded_Y)


# In[218]:


#build the model
model=Sequential()
model.add(Dense(100,input_dim=16,activation="relu"))
model.add(Dense(500,activation="relu"))
model.add(Dense(200,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(7,activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(aria_svd,dummy_y,nb_epoch=20, batch_size=5,verbose=0)




# In[219]:


predictions = model.predict(aria_svd)
positions=np.argmax(predictions, axis=1)
predictions_onehot=list()
for i in positions:
    zeros=[0,0,0,0,0,0,0]
    zeros[i]=1
    predictions_onehot.append(zeros)
n=0
for i,j in zip(predictions_onehot,dummy_y.tolist()):
    if i==j:
        n+=1
n
#n=1919, so the accuracy is 1919/2204=87.06%

# In[205]:


#create the TF-IDF vector of dev set
df_dev= pd.read_csv("ariaset_dev.tsv",sep="\t",encoding="utf-8",names=columns)
df_dev=df_dev.drop(['?'], axis=1)
df_dev=df_dev.dropna()
df_dev=df_dev.sample(frac=1)
dev_text=df_dev["Text"]
dev_emotion=df_dev["Emotion"]
dev_emotionlist=dev_emotion.tolist()
encoder=LabelEncoder()
encoder.fit(dev_emotionlist)
encoded_dev=encoder.transform(dev_emotionlist)
dummy_dev=np_utils.to_categorical(encoded_dev)
vectorizer = TfidfVectorizer(stop_words={"italian"})
dev_tfidf = vectorizer.fit_transform(dev_text)
#create the topic vectors using truncated SVD
dev_svd=svd.fit_transform(dev_tfidf)
dev_svd=pd.DataFrame(dev_svd)
sc=StandardScaler()
dev_svd=sc.fit_transform(dev_svd)
dev_svd.shape
#Now we have dev_svd and dummy_dev


# In[220]:


predictions = model.predict(dev_svd)
positions=np.argmax(predictions, axis=1)
predictions_onehot=list()
for i in positions:
    zeros=[0,0,0,0,0,0,0]
    zeros[i]=1
    predictions_onehot.append(zeros)
n=0
for i,j in zip(predictions_onehot,dummy_dev.tolist()):
    if i==j:
        n+=1
n

#n=53, 53/248=21%  The performance of this model is really not satisfying......
