from Preprocessing import *
import it_core_news_sm
import re
nlp = it_core_news_sm.load()
def tokenizer_FASTTEXT(doc):
    tokenize = []
    new_verse = []
    for x in doc:
        verse = nlp(x)
        new_verse = []
        for w in verse:
            regex = re.compile(r'( +|\'|\-|\,|\!|\:|\;|\?|\.|\(|\)|\«|\»|\")')
            if not regex.match(w.text):
                w_lower = w.text.casefold()
                new_verse.append(w_lower)
        tokenize.append(" ".join(new_verse))

    return tokenize


df_train = pd.concat([cv_text, dev_text])
train_emotion = np.concatenate([emotion, dev_emotion])


cv_tokenized = tokenizer_FASTTEXT(cv_text)
dev_tokenized = tokenizer_FASTTEXT(dev_text)
test_tokenized = tokenizer_FASTTEXT(test_text)
train_tokenized = tokenizer_FASTTEXT(df_train)


#prepare dataset for fasttext


cv = []
for i, j in zip(emotion, cv_tokenized):
    t = "__label__"+i+" "+j+"\n"
    cv.append(t)
file_train = open("cv.txt", "w",encoding="latin-1")
file_train.writelines(cv)
#with open("cv.txt", 'w',encoding="latin-1") as f:
#    for s in cv:
#        f.write(str(s))

dev = []
for i, j in zip(dev_emotion, dev_tokenized):
    t = "__label__"+i+" "+j+"\n"
    dev.append(t)
file_train = open("dev.txt", "w",encoding="latin-1")
file_train.writelines(dev)
dev_text_fa = label_data_return_list(dev_emotion, dev_tokenized)

test = []
for i, j in zip(test_emotion, test_tokenized):
    t = "__label__"+i+" "+j+"\n"
    test.append(t)
file_train = open("test.txt", "w",encoding="latin-1")
file_train.writelines(test)
test_text_fa = label_data_return_list(test_emotion, test_tokenized)

train = []
for i, j in zip(train_emotion, train_tokenized):
    t = "__label__"+i+" "+j+"\n"
    train.append(t)
file_train = open("train.txt", "w",encoding="latin-1")
file_train.writelines(train)
