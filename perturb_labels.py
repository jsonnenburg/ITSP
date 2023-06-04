import pandas as pd
import numpy as np
import random
import os
import argparse
import ast


# perturbation functions #
def perturb_label_uniform(label, perturb_share, all_types):
    """
    Function to perturb labels uniformly, not taking into account pre-perturbation label distributions.

    :param label: The BIO label for an entity.
    :param perturb_share: The share of perturbed labels.
    :param all_types: The set of all occurring BIO label types for the dataset at hand.
    :return: The (perturbed) label.
    """
    all_types_list = list(all_types)
    if np.random.sample(1) < perturb_share:
        return random.sample(all_types_list, 1)[0]
    else:
        return label


def perturb_label_list(label_list, perturb_share, all_types, perturb_function):
    """
    Wrapper function for the perturbation functions, applies the function to a list of labels.
    """
    return [perturb_function(label, perturb_share, all_types) for label in label_list]


def main():
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--dataset_dir",
                        default="COVIDNEWS",
                        type=str,
                        help="the input dataset directory.")
    #parser.add_argument("--column_converters",
    #                    default={'sequence_tok': pd.eval, 'ner_BIO_full': pd.eval},
    #                    help="the converters for the string columns that contain lists")
    parser.add_argument("--perturb_function",
                        default=perturb_label_uniform)

    args = parser.parse_args()

    # read in data
    print(os.path.join('../data', args.dataset_dir, 'data_train.csv'))
    data_train = pd.read_csv(os.path.join('../data', args.dataset_dir, 'data_train.csv'), index_col=0)

    # convert last two cols to lists
    last_two_columns = data_train.columns[-2:]
    for column in last_two_columns:
        data_train[column] = data_train[column].astype(str)
        data_train[column] = data_train[column].apply(lambda x: ast.literal_eval(x))

    # get set of all labeled entity types
    ALL_TYPES = set(" ".join(data_train["labels"].apply(lambda x: x.strip("\n"))).split(" "))

    perturb_shares = [0.05, 0.1, 0.15, 0.2, 0.25]  # List of perturbation shares

    output_dir = os.path.join('../data', args.dataset_dir)
    os.makedirs(output_dir, exist_ok=True)

    for perturb_share in perturb_shares:
        # Apply perturbation to labels
        random.seed(123)  # Add seed for reproducibility
        data_train["ner_BIO_full" + "_" + str(perturb_share)] = data_train["ner_BIO_full"].apply(
            perturb_label_list, perturb_share=perturb_share, all_types=ALL_TYPES, perturb_function=eval(args.perturb_function))

    # Generate statistics of label distributions
    label_stats = data_train["ner_BIO_full"].apply(pd.Series.value_counts)

    # Print and save label distribution statistics
    label_stats_filename = os.path.join(output_dir, "label_stats.txt")
    with open(label_stats_filename, 'w') as file:
        print(label_stats, file=file)
    print("Label distribution statistics saved to:", label_stats_filename)

    # Save perturbed dataset
    perturbed_data_filename = os.path.join(output_dir, "data_train_perturbed.csv")
    data_train.to_csv(perturbed_data_filename)
    print("Perturbed dataset saved to:", perturbed_data_filename)

if __name__ == '__main__':
    main()
