import argparse
import numpy as np
from util import parse_sts
from nltk.tokenize import word_tokenize
from sts_metrics import symmetrical_nist, symmetrical_bleu, longest_common, \
    symmetrical_word_error, edit_dist
from scipy.stats import pearsonr


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py

    # Read in the data set texts and labels
    texts, labels = parse_sts(sts_data)

    # Convert the texts to lower case and store in numpy array
    texts = np.array([[text.lower() for text in pair]
                      for pair in texts], dtype=object)

    # Convert the texts to tokens and place into numpy array
    tokens = np.array([[word_tokenize(text) for text in pair]
                       for pair in texts], dtype=object)

    # Convert the labels to a numpy array of floats
    labels = np.array(labels).astype(float)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    # Instantiate a dictionary to store the scores of each metric
    scores = {metric: None for metric in score_types}

    # Define the metrics which require tokens and their function calls as key,
    # value pairs, functions defined in sts_metrics.py sister module
    token_metrics = {
        'NIST': symmetrical_nist,
        'BLEU': symmetrical_bleu,
        'Word Error Rate': symmetrical_word_error,
    }

    # Define the metrics which require strings and their function calls as key,
    # value pairs, functions defined in sts_metrics.py sister module
    string_metrics = {
        'Longest common substring': longest_common,
        'Edit Distance': edit_dist
    }

    # Iterate over all metrics that require tokens
    for metric, function in token_metrics.items():

        # Get the metric score and save the array in the dictionary
        scores[metric] = np.apply_along_axis(function, 1, tokens)

    # Iterate over all metrics that require single string sentences
    for metric, function in string_metrics.items():

        # Get the metric score and save the array in the dictionary
        scores[metric] = np.apply_along_axis(function, 1, texts)

    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:

        # Calculate the pearson correlation coefficient with predicted
        # similarity score and the true (human labeled) similarity score
        score = pearsonr(scores[metric_name], labels)

        # Print the score to stdout via print command
        print(f"{metric_name} correlation: {score[0]:.03f}")


    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)


