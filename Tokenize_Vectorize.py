from pandas import DataFrame

from Preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import it_core_news_sm
import re

nlp = it_core_news_sm.load()
tokenized = []
count = 0
for verse in aria_text:
    doc = nlp(verse)
    tokenized.append([])
    for w in doc:
        tokenized[count].append(w.text)
    count += 1
# here I was checking that the tokenization was alright, there's just a little bit of noise
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
    doc = nlp(verse)  # we could add here .casefold() to make it case insensitive
    for w in doc:
        regex = re.compile(r'( +|\'|\-|\,|\!|\:|\;|\?|\.|\(|\)|\«|\»)')
        if not regex.match(w.text):
            w_lower = w.text.casefold()
            tokenized.append(w_lower)
    return tokenized

#Character 3-grams - Simple TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=1)
trigram_tfidf = vectorizer.fit_transform(raw_documents=aria_text).toarray()
trigram_tfidf_dev = vectorizer.transform(raw_documents=dev_text).toarray()
trigram_tfidf = pd.DataFrame(trigram_tfidf)
trigram_tfidf_dev = pd.DataFrame(trigram_tfidf_dev)
dimensions = trigram_tfidf.shape[1]


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


#Character 3-grams - TruncatedSVD Topic Vectors
svd = TruncatedSVD(n_components=32, n_iter=100)
trigram_svd = svd.fit_transform(trigram_tfidf)
trigram_svd = pd.DataFrame(trigram_svd)
trigram_svd = sc.fit_transform(trigram_svd)

trigram_svd_dev = svd.transform(trigram_tfidf_dev)
trigram_svd_dev = pd.DataFrame(trigram_svd_dev)
trigram_svd_dev = sc.fit_transform(trigram_svd_dev)

aria_svd_x, aria_svd_y, dummy_y_x, dummy_y_y = train_test_split(aria_svd, dummy_y,
                                                                test_size=0.2)  # here I am just splitting the training dataset