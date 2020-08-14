# Aria-classification

## Opera Arie Project (Draft)

### Representation
Baseline: *TF-IDF* vector

**By 16/08**:
1) *TF-IDF* vector (character 3-gram)
2) LSA Topic Vectors (base tfidf)
3) LDiA Topic Vectors (base tfidf)
4) LSA Topic Vectors (3-gram)
5) LDiA Topic Vectors (3-gram) 

### Classification
**(TBD)**:
1) *k-Nearest-Neighbours* 
2) *Multiclass SVM*
3) *NN* 
4) *CNN*


### Questions from 10/08
- Are the verses shuffled or in the original order? 
	- Each aria is identified by an id (e.g. "ZAP1598155_00"), where the last 2 digits indicate verse order. The arias are shuffled, while the verses follow their original order in the aria.
- Should we eliminate unlabelled lines (e.g. "SIVENIO")?
	- These lines may be kept later on but should be ignored in the preprocessing
