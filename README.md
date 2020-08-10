# Aria-classification

## Opera Arie Project (Draft)

### Representation
Baseline: *TF-IDF* vector

**(TBD)**:
1) *TF-IDF* vector (character trigrams)
2) Topic Vectors (*SVD*, *LSA*, *LDiA*)

### Classification
**(TBD)**:
(*k-Nearest-Neighbours* and *NN* are good candidates)


### Questions from 10/08
- Are the verses shuffled or in the original order? 
	- Each aria is identified by an id (e.g. "ZAP1598155_00"), where the last 2 digits indicate verse order. The arias are shuffled, while the verses follow their original order in the aria.
- Should we eliminate unlabelled lines (e.g. "SIVENIO")?
	- These lines may be kept later on but should be ignored in the preprocessing
