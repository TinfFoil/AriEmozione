from pandas import DataFrame

from Preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDiA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import it_core_news_sm
import re

nlp = it_core_news_sm.load()



def italian_tokenizer(verse):
    tokenize = []
    doc = nlp(verse)  # we could add here .casefold() to make it case insensitive
    for w in doc:
        regex = re.compile(r'( +|\'|\-|\,|\!|\:|\;|\?|\.|\(|\)|\«|\»)')
        if not regex.match(w.text):
            w_lower = w.text.casefold()
            tokenize.append(w_lower)

    return tokenize

def tokenizer_FASTTEXT(doc):
    tokenize = []
    for x in doc:
        verse = nlp(x)
        new_verse = []
        for w in verse:
            regex = re.compile(r'( +|\'|\-|\,|\!|\:|\;|\?|\.|\(|\)|\«|\»)')
            if not regex.match(w.text):
                w_lower = w.text.casefold()
                new_verse.append(w_lower)
        tokenize.append(new_verse)

    return tokenize

tokenized = tokenizer_FASTTEXT(aria_text)
tokenized_dev = tokenizer_FASTTEXT(dev_text)

sc = StandardScaler()

#Character 3-grams - Simple TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,3), min_df=1)
trigram_tfidf = vectorizer.fit_transform(raw_documents=aria_text).toarray()
trigram_tfidf_dev = vectorizer.transform(raw_documents=dev_text).toarray()
trigram_tfidf = pd.DataFrame(trigram_tfidf)
trigram_tfidf_dev = pd.DataFrame(trigram_tfidf_dev)



#Words - Simple TF-IDF
vectorizer = TfidfVectorizer(tokenizer=italian_tokenizer)  # Now it seems to be working
aria_tfidf = vectorizer.fit_transform(aria_text).toarray()
aria_tfidf = pd.DataFrame(aria_tfidf)

dev_tfidf = vectorizer.transform(dev_text).toarray()
dev_tfidf = pd.DataFrame(dev_tfidf)





#create the topic vectors using truncated SVD
svd = TruncatedSVD(n_components=32, n_iter=100)
aria_svd = svd.fit_transform(aria_tfidf)
aria_svd = pd.DataFrame(aria_svd)
aria_svd = sc.fit_transform(aria_svd)




dev_svd = svd.transform(dev_tfidf)
dev_svd = pd.DataFrame(dev_svd)
dev_svd = sc.fit_transform(dev_svd)



#Character 3-grams - TruncatedSVD Topic Vectors
svd = TruncatedSVD(n_components=32, n_iter=100)
trigram_svd = svd.fit_transform(trigram_tfidf)
trigram_svd = pd.DataFrame(trigram_svd)
trigram_svd = sc.fit_transform(trigram_svd)

trigram_svd_dev = svd.transform(trigram_tfidf_dev)
trigram_svd_dev = pd.DataFrame(trigram_svd_dev)
trigram_svd_dev = sc.fit_transform(trigram_svd_dev)

#Words - Simple BoW + LDiA
Counter=CountVectorizer(tokenizer=italian_tokenizer)
aria_bow = pd.DataFrame(Counter.fit_transform(raw_documents=aria_text).toarray())
dev_bow = pd.DataFrame(Counter.transform(raw_documents=dev_text).toarray())


ldia = LDiA(n_components=32,learning_method="batch")
aria_ldia = ldia.fit_transform(aria_bow)
aria_ldia = pd.DataFrame(aria_ldia)
aria_ldia = sc.fit_transform(aria_ldia)

dev_ldia = ldia.transform(dev_bow)
dev_ldia = pd.DataFrame(dev_ldia)
dev_ldia = sc.fit_transform(dev_ldia)




#Character Trigrams - Simple BoW + LDiA
Counter = CountVectorizer(analyzer='char', ngram_range=(3,3), min_df=1)
trigram_bow = pd.DataFrame(Counter.fit_transform(raw_documents=aria_text).toarray())
trigram_bow_dev = pd.DataFrame(Counter.transform(raw_documents=dev_text).toarray())

ldia = LDiA(n_components=32,learning_method="batch")
trigram_ldia = ldia.fit_transform(trigram_bow)
trigram_ldia = pd.DataFrame(trigram_ldia)
trigram_ldia = sc.fit_transform(trigram_ldia)

trigram_ldia_dev = ldia.transform(trigram_bow_dev)
trigram_ldia_dev = pd.DataFrame(trigram_ldia_dev)
trigram_ldia_dev = sc.fit_transform(trigram_ldia_dev)

aria_svd_x, aria_svd_y, dummy_y_x, dummy_y_y = train_test_split(aria_svd, dummy_y,
                                                                test_size=0.2)  # here I am just splitting the training dataset

def n_grams_generate(text, text_dev, n): #To be discussed and fixed to include both train and dev
    sc = StandardScaler()
    puliti=list()
    puliti_dev=list()

    print('Cleaning...')
    for i in text:
        pulito = italian_tokenizer(i)
        pulito = "".join(pulito)
        puliti.append(pulito)

    for i in text_dev:
        pulito = italian_tokenizer(i)
        pulito = "".join(pulito)
        puliti_dev.append(pulito)

    print('Calculating TFIDF...')
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n,n), min_df=1)
    ngram_tfidf = vectorizer.fit_transform(raw_documents=puliti).toarray()
    ngram_tfidf = pd.DataFrame(ngram_tfidf)
    ngram_tfidf_dev = vectorizer.transform(raw_documents=puliti_dev).toarray()
    ngram_tfidf_dev = pd.DataFrame(ngram_tfidf_dev)

    Counter = CountVectorizer(analyzer='char', ngram_range=(n,n), min_df=1)
    ngram_bow = pd.DataFrame(Counter.fit_transform(raw_documents=puliti).toarray())
    ngram_bow_dev = pd.DataFrame(Counter.transform(raw_documents=puliti_dev).toarray())

    print('Calculating SVD...')
    svd = TruncatedSVD(n_components=32, n_iter=100)
    ngram_svd = svd.fit_transform(ngram_tfidf)
    ngram_svd = pd.DataFrame(ngram_svd)
    ngram_svd_dev = svd.transform(ngram_tfidf_dev)
    ngram_svd_dev = pd.DataFrame(ngram_svd_dev)

    print('Calculating LDiA...')
    ldia = LDiA(n_components=32, learning_method="batch")
    ngram_ldia = ldia.fit_transform(ngram_bow)
    ngram_ldia = pd.DataFrame(ngram_ldia)
    ngram_ldia_dev = ldia.transform(ngram_bow_dev)
    ngram_ldia_dev = pd.DataFrame(ngram_ldia_dev)

    ngram_svd = sc.fit_transform(ngram_svd)
    ngram_ldia = sc.fit_transform(ngram_ldia)
    ngram_svd_dev = sc.fit_transform(ngram_svd_dev)
    ngram_ldia_dev = sc.fit_transform(ngram_ldia_dev)
    new_dims = ngram_tfidf.shape[1]
    return ngram_tfidf, ngram_svd, ngram_ldia, ngram_tfidf_dev, ngram_svd_dev, ngram_ldia_dev, new_dims

clean_tfidf, clean_svd, clean_ldia, clean_tfidf_dev, clean_svd_dev, clean_ldia_dev, clean_dims = n_grams_generate(aria_text, dev_text, 3)
dimensions = trigram_tfidf.shape[1]

if __name__ == '__main__':
    print(vectorizer.get_feature_names(), f"\nLenght = {len(vectorizer.get_feature_names())}")
    print(vectorizer.vocabulary_, f"\nLenght = {len(vectorizer.vocabulary_)}")
    print('Trigram tfidf shape:', trigram_tfidf.shape)
    print("This is the Train tfidf:\n", aria_tfidf)
    print("This is the Dev tfidf:\n", dev_tfidf)
    print("This is the Train SVD:\n", aria_svd, f"\nShape: {aria_svd.shape}")
    print("This is the Dev SVD:\n", dev_svd, f"\nShape: {dev_svd.shape}")