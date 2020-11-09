
# AriEmozione:  Identifying Emotions in Opera Verses

Our main task was to identify  the  emotion  expressed  in  a  verse,  in the context of an aria.


## Information on Subfolders


1. Subfolder crossvalidation <br />
`This folder contains all the models that perform 10-fold cross validation using the cv dataset and tested on the dev dataset` <br />
2. Subfolder datasets <br />
`This folder contains all datasets that are used in this project. We have a train dataset (ariaset_train.tsv), called "cv" in the code, a development dataset (ariaset_dev.tsv), and a test dataset (ariaset_test.tsv)` <br />
3. Subfolder test<br />
`This folder contains all the models that are trained on the merged cv and dev dataset (ariaset_train.tsv + ariaset_dev.tsv) and tested on the test set`


## Data and Experiments

The corpus used in this study is available [here](https://zenodo.org/record/4022318)

The full batch of results generated by the code is available [here](https://docs.google.com/spreadsheets/d/1Ztjry2mJs6ufCZM1O5CQRyZ8pA5YDnToN0h0NGX1nW0/edit?usp=sharing)


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/Zhangshibf/Aria-emotions.git
    

## Set-up and installation
1. Install [Pandas](https://pandas.pydata.org/) <br />
`pip install pandas` <br />
2. Install [Numpy](https://numpy.org/) <br />
`pip install numpy` <br />
3. Install [Keras](https://keras.io/) <br />
`pip install Keras` <br />
4. Install [sklearn](https://scikit-learn.org/stable/) <br />
`pip install sklearn` <br />
5. Download it_core_news_sm from [spaCy](https://spacy.io/models/it) (necessary for the Italian tokenizer) <br />
`python -m spacy download it_core_news_sm` <br />
6. If you want to run fasttext models: Install [fasttext](https://fasttext.cc/) <br />
`pip install fasttext` <br />
7. If you want to run fasttext models using pre-trained vectors: Download fasttext's Italian [pre-trained vectors](https://fasttext.cc/docs/en/crawl-vectors.html) and save them at the right directory path <br />
`Download the pre-trained Italian vectors from https://fasttext.cc/docs/en/crawl-vectors.html` <br />
`Unzip the file and save it in "D:/vec/cc.it.300.vec/cc.it.300.vec"` <br />


## If you wish to cite our paper:

The link to our paper: (we don't have it yet)

Citation in bib format:

~~~
We don't have one yet.
Citation in bib format is still to be added.
~~~

Full citation:

~~~
We don't have this one either.
Still to be added
~~~
