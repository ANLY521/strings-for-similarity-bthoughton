# Semantic textual similarity using string similarity


This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

## Metrics

This module contains 5 different string similarity metrics as briefly described below.

**NIST :** This algorithm uses the weighted arithmetic mean of n-gram co-occurrence 
in 2-strings to produce the similarity score. For the purposes of this module
the symmetrical NIST score is calculated. 

**BLEU :** BLEU uses the geometric mean of n-gram overlaps to determine the 
similarity score. It should be noted that BLEU does not work as well on sentence
level correlation due in part to the possibility of 0 overlap. For the purposes 
of this module the symmetrical BLEU score is calculated.  

**WER :** WER (Word Error Rate) is calculated by dividing the Levenshtein 
edit-distance score by the number of words in the reference sentence 
(see **Edit Dist** for more information on Levenshtein edit-distance). For the 
purposes of this module the symmetrical Word Error Rate is calculated.  

**LCS :** LCS (Longest Common Substring) determines the length of the longest 
substring which exist in both the hypothesis and reference. For example the LCS
of the following two strings "abc cat" and "hello abc dog" would be 4, including 
the white space character.

**Edit Distance :** This is the Levenshtein edit-distance algorithm. It
calculates the minimum number of edits to make the reference and hypothesis
identical at the character level. There are 3 possible character operations, 
substitution, insertion and deletion. 


## Correlations

Metric | Train  | Dev    | Test 
-------|--------|--------|-----
NIST | 0.496  | 0.593  | 0.475
BLEU | 0.371  | 0.433  | 0.353
WER | -0.353 | -0.452 | -0.358
LCS | 0.362  | 0.468  | 0.347
Edit Dist | 0.033  | -0.175 | -0.039


## Usage Example

`python sts_pearson.py --sts_data stsbenchmark/sts-dev.csv`

The argument following `--sts-data` flag can be any of the following:

`sts-dev.csv`
`sts-train.csv`
`sts-test.csv`

## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.