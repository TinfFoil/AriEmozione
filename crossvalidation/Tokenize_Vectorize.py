from Preprocessing_Pipeline.Preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDiA
from sklearn.preprocessing import StandardScaler
import it_core_news_sm
import re

sc = StandardScaler()
nlp = it_core_news_sm.load()  # the SpaCy Italian Tokenizer


def italian_tokenizer(verse):
    tokenized = []
    doc = nlp(verse)  # we could add here .casefold() to make it case insensitive
    for w in doc:
        regex = re.compile(r'( +|\'|\-|\,|\!|\:|\;|\?|\.|\(|\)|\«|\»|\")')
        if not regex.match(w.text):
            w_lower = w.text.casefold()
            tokenized.append(w_lower)
    return tokenized





# Words - TF-IDF
vectorizer = TfidfVectorizer(tokenizer=italian_tokenizer)
cv_tfidf = vectorizer.fit_transform(cv_text).toarray()
cv_tfidf = pd.DataFrame(cv_tfidf)

dev_tfidf = vectorizer.transform(dev_text).toarray()
dev_tfidf = pd.DataFrame(dev_tfidf)

vectorizer2 = TfidfVectorizer(tokenizer=italian_tokenizer)
train_tfidf = vectorizer2.fit_transform(df_train).toarray()
train_tfidf = pd.DataFrame(train_tfidf)

test_tfidf = vectorizer2.transform(test_text).toarray()
test_tfidf = pd.DataFrame(test_tfidf)

# Words - SVD
svd = TruncatedSVD(n_components=32, n_iter=100)
cv_svd = svd.fit_transform(cv_tfidf)
cv_svd = pd.DataFrame(cv_svd)
cv_svd = sc.fit_transform(cv_svd)

dev_svd = svd.transform(dev_tfidf)
dev_svd = pd.DataFrame(dev_svd)
dev_svd = sc.fit_transform(dev_svd)

svd2 = TruncatedSVD(n_components=32, n_iter=100)

train_svd = svd2.fit_transform(train_tfidf)
train_svd = pd.DataFrame(train_svd)
train_svd = sc.fit_transform(train_svd)

test_svd = svd2.transform(test_tfidf)
test_svd = pd.DataFrame(test_svd)
test_svd = sc.fit_transform(test_svd)

# Words - LDiA
Counter = CountVectorizer(tokenizer=italian_tokenizer)
cv_bow = pd.DataFrame(Counter.fit_transform(raw_documents=cv_text).toarray())
dev_bow = pd.DataFrame(Counter.transform(raw_documents=dev_text).toarray())

ldia = LDiA(n_components=32, learning_method="batch")
cv_ldia = ldia.fit_transform(cv_bow)
cv_ldia = pd.DataFrame(cv_ldia)
cv_ldia = sc.fit_transform(cv_ldia)

dev_ldia = ldia.transform(dev_bow)
dev_ldia = pd.DataFrame(dev_ldia)
dev_ldia = sc.fit_transform(dev_ldia)

Counter2 = CountVectorizer(tokenizer=italian_tokenizer)

train_bow = pd.DataFrame(Counter2.fit_transform(raw_documents=df_train).toarray())
test_bow = pd.DataFrame(Counter2.transform(raw_documents=test_text).toarray())

ldia2 = LDiA(n_components=32, learning_method="batch")

train_ldia = ldia2.fit_transform(train_bow)
train_ldia = pd.DataFrame(train_ldia)
train_ldia = sc.fit_transform(train_ldia)

test_ldia = ldia2.transform(test_bow)
test_ldia = pd.DataFrame(test_ldia)
test_ldia = sc.fit_transform(test_ldia)

# Character 3-grams - TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1)

trigram_tfidf_cv = vectorizer.fit_transform(raw_documents=cv_text).toarray()
trigram_tfidf_cv = pd.DataFrame(trigram_tfidf_cv)

trigram_tfidf_dev = vectorizer.transform(raw_documents=dev_text).toarray()
trigram_tfidf_dev = pd.DataFrame(trigram_tfidf_dev)

vectorizer2 = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1)

trigram_tfidf_train = vectorizer2.fit_transform(raw_documents=df_train).toarray()
trigram_tfidf_test = vectorizer2.transform(raw_documents=test_text).toarray()
trigram_tfidf_train = pd.DataFrame(trigram_tfidf_train)
trigram_tfidf_test = pd.DataFrame(trigram_tfidf_test)

# Character 3-grams - SVD
svd = TruncatedSVD(n_components=32, n_iter=100)

trigram_svd_cv = svd.fit_transform(trigram_tfidf_cv)
trigram_svd_cv = pd.DataFrame(trigram_svd_cv)
trigram_svd_cv = sc.fit_transform(trigram_svd_cv)

trigram_svd_dev = svd.transform(trigram_tfidf_dev)
trigram_svd_dev = pd.DataFrame(trigram_svd_dev)
trigram_svd_dev = sc.fit_transform(trigram_svd_dev)

svd2 = TruncatedSVD(n_components=32, n_iter=100)

trigram_svd_train = svd2.fit_transform(trigram_tfidf_train)
trigram_svd_train = pd.DataFrame(trigram_svd_train)
trigram_svd_train = sc.fit_transform(trigram_svd_train)

trigram_svd_test = svd2.transform(trigram_tfidf_test)
trigram_svd_test = pd.DataFrame(trigram_svd_test)
trigram_svd_test = sc.fit_transform(trigram_svd_test)

# Character Trigrams - Simple BoW + LDiA
Counter = CountVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1)

trigram_bow_cv = pd.DataFrame(Counter.fit_transform(raw_documents=cv_text).toarray())
trigram_bow_dev = pd.DataFrame(Counter.transform(raw_documents=dev_text).toarray())

ldia = LDiA(n_components=32, learning_method="batch")
trigram_ldia_cv = ldia.fit_transform(trigram_bow_cv)
trigram_ldia_cv = pd.DataFrame(trigram_ldia_cv)
trigram_ldia_cv = sc.fit_transform(trigram_ldia_cv)

trigram_ldia_dev = ldia.transform(trigram_bow_dev)
trigram_ldia_dev = pd.DataFrame(trigram_ldia_dev)
trigram_ldia_dev = sc.fit_transform(trigram_ldia_dev)

Counter2 = CountVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1)

trigram_bow_train = pd.DataFrame(Counter2.fit_transform(raw_documents=df_train).toarray())
trigram_bow_test = pd.DataFrame(Counter2.transform(raw_documents=test_text).toarray())

ldia = LDiA(n_components=32, learning_method="batch")

trigram_ldia_train = ldia.fit_transform(trigram_bow_train)
trigram_ldia_train = pd.DataFrame(trigram_ldia_train)
trigram_ldia_train = sc.fit_transform(trigram_ldia_train)

trigram_ldia_test = ldia.transform(trigram_bow_test)
trigram_ldia_test = pd.DataFrame(trigram_ldia_test)
trigram_ldia_test = sc.fit_transform(trigram_ldia_test)

# Dimensions of the various TFIDF vectors
dim_cv_char = trigram_tfidf_cv.shape[1]
dim_cv_word = cv_tfidf.shape[1]
dim_train_char = trigram_tfidf_train.shape[1]
dim_train_word = cv_tfidf.shape[1]

if __name__ == "__main__":
    print(vectorizer.get_feature_names(), f"\nLength = {len(vectorizer.get_feature_names())}")
    print(vectorizer.vocabulary_, f"\nLength = {len(vectorizer.vocabulary_)}")

    print("This is the Train tfidf:\n", train_tfidf)
    print("This is the Test tfidf:\n", test_tfidf)
    print("This is the Train SVD:\n", train_svd, f"\nShape: {train_svd.shape}")
    print("This is the Test SVD:\n", test_svd, f"\nShape: {test_svd.shape}")
