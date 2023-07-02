import pandas as pd
import argparse
import os
from nltk.tokenize import word_tokenize

"""
(Archived / superseded by generation of labels from conllu output file)
Generates entity labels for sequences from skweak entity dictionary.
"""


def join_strings(row):
    return ' '.join(row)


def transform_sequences(labeled_sequences):
    sequences = []
    labels = []
    sequence_tok = []
    labels_tok = []

    for line in labeled_sequences:
        if line == '\n':
            sequences.append(sequence_tok)
            labels.append(labels_tok)
            sequence_tok = []
            labels_tok = []
        else:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                token, label = parts
            else:
                token = parts[0]
                label = 'O'
            sequence_tok.append(token)
            labels_tok.append(label)

    df = pd.DataFrame({'sequence_tok': sequences, 'labels_tok': labels})

    return df


def main():
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--dataset_dir",
                        default="COVIDNEWS",
                        type=str,
                        help="the input dataset directory.")

    args = parser.parse_args()

    # read in data
    file_path = os.path.join('../data', args.dataset_dir, 'dist_skweak/skweak_output.csv')
    train = open(file_path, 'r').readlines()

    output_dir = os.path.join('../data', args.dataset_dir, 'dist_skweak')

    data_train = transform_sequences(train)

    data_train['labels'] = data_train['labels_tok'].apply(lambda x: ' '.join(x))
    data_train['sequence'] = data_train['sequence_tok'].apply(lambda x: ' '.join(x))

    # reorder columns
    data_train = data_train[['sequence', 'sequence_tok', 'labels', 'labels_tok']]

    data_train.to_csv(os.path.join(output_dir, "data_train.csv"))
    print("Saved dataset to:", os.path.join(output_dir, "data_train.csv"))


if __name__ == '__main__':
    main()
