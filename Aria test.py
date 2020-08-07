#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
columns=["Names","Text","Emotion","Fiducia","?"]
df = pd.read_csv("ariaset_train.tsv",sep="\t",encoding="utf-8",names=columns)
aria_text=df["Text"]


# In[35]:


from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
tokens=list()
for i in aria_text:
    tokens.append([tokenizer.tokenize(i.lower())])


# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
vectorizer = TfidfVectorizer()
aria_tfidf = vectorizer.fit_transform(aria_text)
print(aria_tfidf.shape)
svd=TruncatedSVD(n_components=16,n_iter=100)
aria_svd=svd.fit_transform(aria_tfidf)
aria_svd=pd.DataFrame(aria_svd)
aria_svd.round(3)


# In[ ]:


"""1. How does multi-label classification with LDA work? 
2. Are those texts shuffled or in the original order?
3. Should we eliminate lines like ALESSANDRO?
"""


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
X_train,X_test,y_train,y_test=()

