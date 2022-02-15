# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """

    # Instantiate empty lists to store the text and labels
    texts = []
    labels = []

    # Open the sts data file
    with open(data_file, 'r', encoding='utf-8') as f:

        # Iterate over each line in the data set
        for line in f:

            # Split the line into features (columns)
            fields = line.strip().split('\t')

            # Add the labels to the list
            labels.append(fields[4])

            # Get the texts as lower case and remove new line character
            t1 = fields[5].lower().replace('\n', '')
            t2 = fields[6].lower().replace('\n', '')

            # Add the 2 texts as a tuple to the list
            texts.append((t1, t2))

        # Close the file connection
        f.close()

    return texts, labels


if __name__ == '__main__':

    # Test the parse sts function
    txt, lbl = parse_sts('stsbenchmark/sts-dev.csv')
    print(txt[0], lbl[0])
