# Crossvalidation subfolder

This folder contains the models for the 10-fold cross validation using the cv dataset and testing on the dev dataset.

## Description of the files
- ```Neuron_Epochs_Exploration``` directory containing the neural network pipelines to find the best neuron and epoch combinations.

- ```Fasttext_no_pretrained_vec_crossval_.py``` Fasttext model without any preloaded word vectors.

- ```LogReg_crossval_.py``` Logistic Regression

- ```SVM_crossval_.py``` Support Vector Machine

- ```ThreeNN_crossval_.py``` Three-layer Neural Network

- ```TwoNN_crossval_.py``` Two-layer Neural Network

- ```kNN_crossval_.py``` k-Nearest-Neighbours

## How to run the code
1. Add the desired model and ALL the contents of ```preprocessing_pipeline``` into a directory of your choice.
2. Add the AriEmozione corpus into the same directory.
3. Run the code either directly from your preferred IDE or via the command line using:
```
$ python foo.py
```


### WARNING
If you wish to rerun the whole pipeline to find new Neuron/Epochs combinations for ```ThreeNN_crossval_.py``` and ```TwoNN_crossval_.py```, first make sure you load the respective pipelines from ```Neuron_Epochs_Exploration``` in your directory, namely ```Best_Neuron_Epochs3.py``` and ```Best_Neuron_Epochs2.py```.


Afterwards, change as follows the first line of code for either file to ensure the imports are correct.


For ```ThreeNN_crossval_.py``` you should change:


 ```from Tokenize_Vectorize import *```  to ```from Best_Neuron_Epochs3 import *``` 

Whereas for ```TwoNN_crossval_.py``` you should change:


```from Tokenize_Vectorize import *```  to ```from Best_Neuron_Epochs2 import *``` 

Once you have done this, you can simply run the script as instructed above.
