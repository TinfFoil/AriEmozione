#!/usr/bin/env python
# coding: utf-8

# In[88]:

import numpy as np
import pandas as pd
columns=["Names","Text","Emotion","Fiducia","?"]
df = pd.read_csv("ariaset_train.tsv",sep="\t",encoding="utf-8",names=columns)
#skipping lines with no label for Emotion
df=df.drop(['?'], axis=1)
df=df.dropna()
df=df.sample(frac=1)
aria_text=df["Text"]
emotion=df["Emotion"]

#creat tf-idf vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
vectorizer = TfidfVectorizer(stop_words={"italian"})
aria_tfidf = vectorizer.fit_transform(aria_text)

#create the topic vectors using truncated SVD
svd=TruncatedSVD(n_components=16,n_iter=100)
aria_svd=svd.fit_transform(aria_tfidf)
aria_svd=pd.DataFrame(aria_svd)
aria_svd.round(3)

#LDiA topic model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
Counter=CountVectorizer(tokenizer=casual_tokenize)
bow_docs=pd.DataFrame(Counter.fit_transform(raw_documents=aria_text).toarray())

# In[85]:


#use LDiA to creat topic vectors
from sklearn.decomposition import LatentDirichletAllocation as LDiA
ldia=LDiA(n_components=16,learning_method="batch")
ldia=ldia.fit(bow_docs)
ldia_topic_vector=ldia.transform(bow_docs)
ldia_tv=pd.DataFrame(ldia_topic_vector)


# In[87]:


ldia_tv.round(2)


#Character 3-grams - Simple TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=1)
trigram_tfidf = vectorizer.fit_transform(raw_documents=aria_text).toarray()
trigram_tfidf = pd.DataFrame(trigram_tfidf)

#Character 3-grams - Simple BoW
Counter = CountVectorizer(analyzer='char', ngram_range=(3,3), min_df=1)
trigram_bow = Counter.fit_transform(raw_documents=aria_text)
trigram_bow = pd.DataFrame(trigram_bow).toarray()

#Character 3-grams - TruncatedSVD Topic Vectors
svd = TruncatedSVD(n_components=16, n_iter=100)
trigram_svd = svd.fit_transform(trigram_tfidf)
trigram_svd = pd.DataFrame(trigram_svd)

#Character 3-grams - PCA Topic Vectors
pca = PCA(n_components=16)
trigram_pca = pca.fit_transform(trigram_tfidf)
trigram_pca = pd.DataFrame(trigram_pca)

#Character 3-grams - LDiA Topic Vectors
ldia = LDiA(n_components=16, learning_method="batch")
trigram_ldia = ldia.fit_transform(trigram_bow)
trigram_ldia = pd.DataFrame(trigram_ldia)


