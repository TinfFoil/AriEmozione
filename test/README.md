# Test subfolder

This folder contains the models trained on the merged train+dev dataset and tested on the test dataset.


## Description of the files
- ```Fasttext_no_pretrained_vec_test_.py``` Fasttext model without any preloaded word vectors.

- ```Fasttext_pretrained_vec_test_.py``` Fasttext model with the Fasttext word vectors.

- ```LogReg_test_.py``` Logistic Regression

- ```SVM_test_.py``` Support Vector Machine

- ```ThreeNN_test_.py``` Three-layer Neural Network

- ```TwoNN_test_.py``` Two-layer Neural Network

- ```kNN_test_.py``` k-Nearest-Neighbours


## How to run the code
1. Add the desired model and ALL the contents of ```preprocessing_pipeline``` into a directory of your choice.
2. Add the AriEmozione corpus into the same directory.
3. Run the code either directly from your preferred IDE or via the command line using:
```
$ python foo.py
```
