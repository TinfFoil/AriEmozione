# Preprocessing subfolder

This folder contains the preprocessing pipeline to be used in both crossvalidation and test.

## Files description:
- ```Fasttext_preprocessing.py``` An additional specific preprocessing pipeline for Fasttext, which only accepts specific labels.

- ```Processing.py``` The preprocessing pipeline for all models. It also extracts and stores the labels from the training set as both one-hot ```encoded_foo```  and categoricals ```foo_y```. 

- ```Tokenize_Vectorize.py``` Pipeline for the tokenization and vectorization of all models.

## How to run the code
If you wish to run these pipelines to have a look at the various representations:
1) Add the desired model a directory of your choice.
2) Add the AriEmozione corpus into the same directory.
3) Run the code either directly from your preferred IDE or via the command line using:
```
$ python foo.py
```
