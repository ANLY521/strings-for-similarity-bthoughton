import argparse
import numpy as np
from util import parse_sts
from nltk.tokenize import word_tokenize
from sts_metrics import symmetrical_nist, symmetrical_bleu, longest_common, \
    symmetrical_word_error, edit_dist
from scipy.stats import pearsonr


def generate_scores(
        current_text: tuple,
        token_metrics: dict,
        string_metrics: dict,
        scores: dict
) -> None:
    """
    Calculates and stores the score of various string similarity metrics.

    Arguments:
        current_text (tuple): The sentences to compare. Represented as a
            tuple containing with a single string for each sentence, i.e.
            ("This is sentence one.", "Here is sentence 2.").
        token_metrics (dict): The metrics which require the texts to be
            tokenized, with keys as the metric names and values as the function
            to calculate the score.
        string_metrics (dict): The metrics which require the texts to be
            represented as single/complete strings, with keys as the metric
            names and values as the function to calculate the score.
        scores (dict):
            The score dictionary to update for each metric.

    """

    # Define the texts to be compared
    text1, text2 = current_text[0].lower(), current_text[1].lower()

    text1 = text1.lower()
    text2 = text2.lower()

    # Tokenize the texts
    t1_tokens = word_tokenize(text1)
    t2_tokens = word_tokenize(text2)

    # Iterate over all metrics that require tokens
    for metric, method in token_metrics.items():

        # Get the metric score
        metric_score = method((t1_tokens, t2_tokens))

        # Add the score to the score dictionary for current metric and texts
        scores[metric].append(metric_score)

    # Iterate over all metrics that require single string sentences
    for metric, method in string_metrics.items():

        # Get the score in each order
        metric_score = method((text1, text2))

        # Add the score to the score dictionary for current metric and texts
        scores[metric].append(metric_score)


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py

    # Read in the data set texts and labels
    texts, labels = parse_sts(sts_data)

    # Convert the texts to lower case and store in numpy array
    texts = np.array([[text.lower() for text in pair] for pair in texts], dtype=object)

    # Convert the texts to tokens and place into numpy array
    tokens = np.array([[word_tokenize(text) for text in pair] for pair in texts], dtype=object)

    # Convert the labels to a numpy array of floats
    labels = np.array(labels).astype(float)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    # Instantiate a dictionary to store the scores of each metric
    scores = {metric: [] for metric in score_types}

    # Define the token metrics and their function calls as key, value pairs
    token_metrics = {
        'NIST': symmetrical_nist,
        'BLEU': symmetrical_bleu,
        'Word Error Rate': symmetrical_word_error,
    }

    # Define the string metrics and their function calls as key, value pairs
    string_metrics = {
        'Longest common substring': longest_common,
        'Edit Distance': edit_dist
    }

    # Iterate over all texts in the corpus
    for text_pair in texts:

        # Generate the score for each metric for the current text pair
        generate_scores(text_pair, token_metrics, string_metrics, scores)

    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:

        if metric_name in []:
            pass

        else:
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


