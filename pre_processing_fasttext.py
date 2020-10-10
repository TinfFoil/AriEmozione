def label_data_return_list(results, data):
    labelle = list()
    for i, j in zip(results, data):
         = "__label__"+i+" "+j+"\n"
        labelled.append(t)
    return labelled
def convert_pre(pre):
    y_pre = list()
    for r in pre:
        if r[0][0 = ='__label__Ammirazione':
            y_pred.append(0)
        if r[0][0 = ='__label__Amore':
            y_pred.append(1)
        if r[0][0 = ='__label__Gioia':
            y_pred.append(2)
        if r[0][0 = ='__label__Paura':
            y_pred.append(3)
        if r[0][0 = ='__label__Rabbia':
            y_pred.append(4)
        if r[0][0 = ='__label__Tristezza':
            y_pred.append(5)
    return y_pred

def label_data(results, data, filename):
    labelle = list()
    for i, j in zip(results, data):
         = "__label__"+i+" "+j+"\n"
        labelled.append(t)
    f = filename+".txt"
    f = open(fn, "w")
    fi.writelines(labelled)
    return fn

def convert_emotion_list_to_string_of_numbers(emotion):
    emotion_list = emotion.tolist()
    encoder = LabelEncoder()
    encoder.fit(emotion_list)
    encode = encoder.transform(emotion_list)
    return encoded


from functools import reduce
  
def Average(lst): 
    return reduce(lambda a, b: a + b, lst) / len(lst) 


nlp = it_core_news_sm.load()
def tokenizer_FASTTEXT(doc):
    tokenize = []
    new_vers = []
    for x in doc:
        verse = nlp(x)
        new_verse = []
        for w in verse:
            regex = re.compile(r'( +|\'|\-|\, |\!|\:|\;|\?|\.|\(|\)|\«|\»)')
            if not regex.match(w.text):
                w_lower = w.text.casefold()
                new_verse.append(w_lower)
        tokenize.append(" ".join(new_verse))

    return tokenize

train_dev_tex = pd.concat([aria_text, dev_text])
train_dev_emotio = np.concatenate([emotion, dev_emotion])


train_tokenized = tokenizer_FASTTEXT(aria_text)
dev_tokenized = tokenizer_FASTTEXT(dev_text)
test_tokenized = tokenizer_FASTTEXT(test_text)
train_dev_tokenized = tokenizer_FASTTEXT(train_dev_text)

#prepare dataset for fasttext
trai = []
for i, j in zip(emotion, train_tokenized):
     = "__label__"+i+" "+j+"\n"
    train.append(t)
file_trai = open("train.txt", "w")
file_train.writelines(train)

de = []
for i, j in zip(dev_emotion, dev_tokenized):
     = "__label__"+i+" "+j+"\n"
    dev.append(t)
file_trai = open("dev.txt", "w")
file_train.writelines(dev)
dev_text_f = label_data_return_list(dev_emotion, dev_tokenized)

tes = []
for i, j in zip(test_emotion, test_tokenized):
     = "__label__"+i+" "+j+"\n"
    test.append(t)
file_trai = open("test.txt", "w")
file_train.writelines(test)
test_text_f = label_data_return_list(test_emotion, test_tokenized)

train_de = []
for i, j in zip(train_dev_emotion, train_dev_tokenized):
     = "__label__"+i+" "+j+"\n"
    train_dev.append(t)
file_trai = open("train_dev.txt", "w")
file_train.writelines(train_dev)

