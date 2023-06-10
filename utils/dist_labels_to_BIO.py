import pandas as pd
import os
import ast
import argparse


def generate_bio_labels(entity_labels):
    bio_labels = []
    for label in entity_labels:
        if label == 'O':
            bio_labels.append(label)
        else:
            if bio_labels and bio_labels[-1] == label:
                bio_labels.append('I-' + label.lower())
            else:
                bio_labels.append('B-' + label.lower())
    return bio_labels


def main():
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--dataset_dir",
                        default="COVIDNEWS",
                        type=str,
                        help="the input dataset directory.")
    parser.add_argument("--sampling_method",
                        default="strat",
                        type=str,
                        help="the sampling method (unif, strat,...).")

    args = parser.parse_args()

    # read in data
    data_train = pd.read_csv(os.path.join('../data', args.dataset_dir,
                                          f'data_perturbed/raw/data_train_perturbed_{args.sampling_method}.csv'),
                             index_col=0)

    output_dir = os.path.join('../data', args.dataset_dir, f'data_perturbed/BIO')
    os.makedirs(output_dir, exist_ok=True)

    relev_cols = [col for col in data_train if col.startswith('labels_tok')]

    for column in relev_cols:
        data_train[column] = data_train[column].astype(str)
        data_train[column] = data_train[column].apply(lambda x: ast.literal_eval(x))

        data_train[column] = data_train[column].apply(lambda x: generate_bio_labels(x))

    data_train.to_csv(os.path.join(output_dir, f"data_train_perturbed_{args.sampling_method}.csv"))
    print("Saved dataset to:", os.path.join(output_dir, f"data_train_perturbed_{args.sampling_method}.csv"))


if __name__ == '__main__':
    main()
