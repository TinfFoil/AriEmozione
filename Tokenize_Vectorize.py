from Preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    doc = nlp(verse)   #we could add here .casefold() to make it case insensitive
    for w in doc:
        tokenized.append(w.text)
    return tokenized


vectorizer = TfidfVectorizer(tokenizer=italian_tokenizer)  # Now it seems to be working
aria_tfidf = vectorizer.fit_transform(aria_text).toarray()
dev_tfidf = vectorizer.transform(dev_text).toarray()
aria_tfidf = pd.DataFrame(aria_tfidf)
dev_tfidf = pd.DataFrame(dev_tfidf)


sc = StandardScaler()
#create the topic vectors using truncated SVD
svd = TruncatedSVD(n_components=32, n_iter=100)
aria_svd = svd.fit_transform(aria_tfidf)
aria_svd = pd.DataFrame(aria_svd)
aria_svd = sc.fit_transform(aria_svd)



dev_svd = svd.transform(dev_tfidf)
dev_svd = pd.DataFrame(dev_svd)
dev_svd = sc.fit_transform(dev_svd)


aria_svd_x, aria_svd_y, dummy_y_x, dummy_y_y = train_test_split(aria_svd, dummy_y,
                                                                test_size=0.2)  # here I am just splitting the training dataset