from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
import argparse
import numpy as np


def symmetrical_nist(texts):
    """
    Calculates symmetrical similarity as NIST(a,b) + NIST(b,a).

    Arguments:
        texts (tuple): The sentences to compare. Represented as a tuple
        containing with a single string for each sentence, i.e.
        ("This is sentence one.", "Here is sentence 2.").

    Returns:
        nist_score (float): The symmetrical NIST score of the 2 sentences.
    """

    text1, text2 = texts

    # input tokenized text
    t1_tokens = word_tokenize(text1.lower())
    t2_tokens = word_tokenize(text2.lower())

    # Try to calculate the first score, zero division is possible due to
    # the lowest score being 0
    try:
        nist_1 = sentence_nist([t1_tokens, ], t2_tokens)

    # If zero division occurs raise error and set first score to 0
    except ZeroDivisionError as err:
        print('No NIST', err)
        # Set the score to 0
        nist_1 = 0.0

    # Try to calculate the second score, zero division is possible due to
    # the lowest score being 0
    try:
        nist_2 = sentence_nist([t2_tokens, ], t1_tokens)

    # If zero division occurs raise error and set second score to 0
    except ZeroDivisionError as err:
        print('No NIST', err)
        # Set the score to 0
        nist_2 = 0.0

    # Add scores to get final score value
    nist_score = nist_1 + nist_2

    return nist_score


def main(sts_data):
    """Calculate NIST metric for pairs of strings
    Data is formatted as in the STS benchmark"""

    # TODO 1: define a function to read the data in util
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # take a sample of sentences so the code runs fast for faster debugging
    # when you're done debugging, you may want to run this on more!
    sample_text = texts[120:140]
    sample_labels = labels[120:140]
    # zip them together to make tuples of text associated with labels
    sample_data = zip(sample_labels, sample_text)

    scores = []
    for label,text_pair in sample_data:
        print(label)
        print(f"Sentences: {text_pair[0]}\t{text_pair[1]}")
        # TODO 2: Calculate NIST for each pair of sentences
        # Define the function symmetrical_nist

        nist_total = symmetrical_nist(text_pair)
        print(f"Label: {label}, NIST: {nist_total:0.02f}\n")
        scores.append(nist_total)

    # This assertion verifies that symmetrical_nist is symmetrical
    # if the assertion holds, execution continues. If it does not, the program crashes
    first_pair = texts[0]
    text_a, text_b = first_pair
    nist_ab = symmetrical_nist((text_a, text_b))
    nist_ba = symmetrical_nist((text_b, text_a))
    assert nist_ab == nist_ba, f"Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}"

    # TODO 3: find and print the sentences from the sample with the highest and lowest scores
    min_score_index = np.argmin(scores)
    min_score = scores[min_score_index]
    print(f"Lowest score: {min_score}")
    print(sample_text[min_score_index])
    assert min_score == symmetrical_nist(sample_text[min_score_index])

    max_score_index = np.argmax(scores)
    max_score = scores[max_score_index]
    print(f"Highest score: {max_score}")
    print(sample_text[max_score_index])
    assert max_score == symmetrical_nist(sample_text[max_score_index])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)
