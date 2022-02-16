from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher
from util import parse_sts
from nltk import edit_distance


def symmetrical_nist(tokens):
    """
    Calculates symmetrical similarity as NIST(a,b) + NIST(b,a).

    Arguments:
        tokens (array): The sentences to compare. Represented as an array
            containing two arrays of the tokenized sentences, where each array
            represents one of the sentences i.e.
            [["this", "is", "sentence", "one"], ["here", "is", "sentence" "2"]].

    Returns:
        nist_score (float): The symmetrical NIST score of the 2 sentences.
    """

    # Define the tokens to compare
    t1_tokens, t2_tokens = tokens

    # Try to calculate the first score, zero division is possible due to
    # the lowest score being 0
    try:
        nist_1 = sentence_nist([t1_tokens, ], t2_tokens)

    # If zero division occurs raise error and set first score to 0
    except ZeroDivisionError as err:
        # print('No NIST', err)
        # Set the score to 0
        nist_1 = 0.0

    # Try to calculate the second score, zero division is possible due to
    # the lowest score being 0
    try:
        nist_2 = sentence_nist([t2_tokens, ], t1_tokens)

    # If zero division occurs raise error and set second score to 0
    except ZeroDivisionError as err:
        # print('No NIST', err)
        # Set the score to 0
        nist_2 = 0.0

    # Add scores to get final score value
    nist_score = nist_1 + nist_2

    return nist_score


def symmetrical_bleu(tokens):
    """
    Calculates symmetrical similarity as BLEU(a,b) + BLEU(b,a).

    Arguments:
        tokens (array): The sentences to compare. Represented as an array
            containing two arrays of the tokenized sentences, where each array
            represents one of the sentences i.e.
            [["this", "is", "sentence", "one"], ["here", "is", "sentence" "2"]].

    Returns:
        bleu_score (float): The symmetrical BLEU score of the 2 sentences.
    """

    # Define the tokens to compare
    t1_tokens, t2_tokens = tokens

    # # Try to calculate the first score, zero division is possible due to
    # # the lowest score being 0
    try:
        bleu_1 = sentence_bleu(
            [t1_tokens, ],
            t2_tokens,
            smoothing_function=SmoothingFunction().method0
        )

    # If zero division occurs raise error and set first score to 0
    except ZeroDivisionError as err:
        print('No BLEU', err)
        # Set the score to 0
        bleu_1 = 0.0

    # Try to calculate the second score, zero division is possible due to
    # the lowest score being 0
    try:
        bleu_2 = sentence_bleu(
            [t2_tokens, ],
            t1_tokens,
            smoothing_function=SmoothingFunction().method0
        )

    # If zero division occurs raise error and set second score to 0
    except ZeroDivisionError as err:
        print('No BLEU', err)
        # Set the score to 0
        bleu_2 = 0.0

    # Add scores to get final score value
    bleu_score = bleu_1 + bleu_2

    return bleu_score


def symmetrical_word_error(tokens):
    """
    Calculates the symmetrical word error rate of two tokenized texts.

    Arguments:
        tokens (array): The sentences to compare. Represented as an array
            containing two arrays of the tokenized sentences, where each array
            represents one of the sentences i.e.
            [["this", "is", "sentence", "one"], ["here", "is", "sentence" "2"]].

    Returns:
        error_rate (float): The symmetrical word error rate score.

    """

    # Define the tokens to compare
    t1_tokens, t2_tokens = tokens

    edit_distance_score = edit_distance(t1_tokens, t2_tokens)

    error_rate = edit_distance_score/len(t1_tokens) + \
        edit_distance_score/len(t2_tokens)

    return error_rate


def longest_common_substring(texts):
    """
    Determines the longest common substring of two texts.

    Arguments:
        texts (array): The sentences to compare. Represented as an array
        containing a single string for each sentence, i.e.
        ["This is sentence one.", "Here is sentence 2."].

    Returns:
        score (int): The length of the longest common substring.
    """

    # Define the tokens to compare
    text1, text2 = texts

    # Instantiate the sequence matcher
    matcher = SequenceMatcher(None, text1, text2)

    # Get the results from the sequence matcher
    results = matcher.find_longest_match(0, len(text1), 0, len(text2))

    # Get the longest common substring value
    score = results[2]

    return score


def edit_dist(texts):
    """
    Calculates the Levenshtein edit-distance between two strings.

    Arguments:
        texts (array): The sentences to compare. Represented as an array
        containing a single string for each sentence, i.e.
        ["This is sentence one.", "Here is sentence 2."]

    Returns:
        score (int): The Levenshtein edit-distance score.
    """

    # Define the tokens to compare
    text1, text2 = texts

    # Calculate the edit distance score
    score = edit_distance(text1, text2)

    return score


if __name__ == '__main__':

    sts_data = 'stsbenchmark/sts-dev.csv'
    txts, lbls = parse_sts(sts_data)
    sim_metrics = SimilarityMetrics(txts, lbls)

    print(sim_metrics.tokens)
