import pandas as pd
import numpy as np
import random
import os
import argparse
import ast
from collections import Counter


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


def perturb_label_stratified(label, perturb_share, data, labels, weights):
    """
    Function to perturb labels using stratified sampling, taking into account pre-perturbation label distributions.

    :param weights:
    :param labels:
    :param label: The BIO label for an entity.
    :param perturb_share: The share of perturbed labels.
    :param data: The dataset.
    :return: The (perturbed) label.
    """

    # TODO
    # put this outside function --> call only once in main script, not for each label

    #all_labs = " ".join(data["labels"].apply(lambda x: x.strip("\n"))).split(" ")

    #c = Counter(all_labs)
    #labels = list(c.keys())
    #weights = list(c.values())

    if np.random.sample(1) < perturb_share:
        return random.choices(labels, weights=weights, k=1)[0]
    else:
        return label


def perturb_label_list_unif(label_list, perturb_share, all_types, perturb_function):
    """
    Wrapper function for the perturbation functions, applies the function to a list of labels.
    """
    return [perturb_function(label, perturb_share, all_types) for label in label_list]


def perturb_label_list_strat(label_list, perturb_share, all_types, labels, weights, perturb_function):
    """
    Wrapper function for the perturbation functions, applies the function to a list of labels.
    """
    return [perturb_function(label, perturb_share, all_types, labels, weights) for label in label_list]


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

    random.seed(123)  # Add seed for reproducibility

    # read in data
    # changed to path to skweak dist labels (non-BIO)
    data_train = pd.read_csv(os.path.join('../data', args.dataset_dir, 'dist_skweak/data_train.csv'), index_col=0)

    # convert last two cols to lists
    data_train['labels_tok'] = data_train['labels_tok'].astype(str)
    data_train['labels_tok'] = data_train['labels_tok'].apply(lambda x: ast.literal_eval(x))

    # get set of all labeled entity types
    all_labs = " ".join(data_train["labels"].apply(lambda x: x.strip("\n"))).split(" ")
    ALL_TYPES = set(all_labs)

    c = Counter(all_labs)
    labels = list(c.keys())
    weights = list(c.values())

    # create label dist stats df
    label_dist_df = pd.DataFrame(c.items(), columns=['Label', 'Count_raw'])

    perturb_shares = [0.05, 0.1, 0.15, 0.2, 0.25]  # List of perturbation shares

    output_dir = os.path.join('../data', args.dataset_dir)
    os.makedirs(output_dir, exist_ok=True)

    for perturb_share in perturb_shares:
        if args.perturb_function == 'perturb_label_uniform':
            data_train["labels_tok" + "_" + "uni" + str(perturb_share)] = data_train["labels_tok"].apply(
                perturb_label_list_unif, perturb_share=perturb_share, all_types=ALL_TYPES,
                perturb_function=eval(args.perturb_function))

            # add column to label dist stats df
            c_temp = Counter(" ".join(data_train["labels_tok" + "_" + "uni" + str(perturb_share)] .sum()).split(" "))
            df_temp = pd.DataFrame(c_temp.items(), columns=['Label', f'Count_uni{str(perturb_share)}'])
            #label_dist_df = pd.concat([label_dist_df, df_temp], axis=1)
            label_dist_df = pd.merge(label_dist_df, df_temp)

            # column names werden übernommen!

        if args.perturb_function == 'perturb_label_stratified':
            data_train["labels_tok" + "_" + "strat" + str(perturb_share)] = data_train["labels_tok"].apply(
                perturb_label_list_strat, perturb_share=perturb_share, all_types=ALL_TYPES, labels=labels,
                weights=weights, perturb_function=eval(args.perturb_function))

            # add column to label dist stats df
            # need to join all labels to then count them
            c_temp = Counter(" ".join(data_train["labels_tok" + "_" + "strat" + str(perturb_share)] .sum()).split(" "))
            df_temp = pd.DataFrame(c_temp.items(), columns=['Label', f'Count_strat{str(perturb_share)}'])
            #label_dist_df = pd.concat([label_dist_df, df_temp], axis=1)
            label_dist_df = pd.merge(label_dist_df, df_temp)

    # individually for each perturb method
    if args.perturb_function == 'perturb_label_uniform':
        label_stats_filename = os.path.join(output_dir, "descriptives/weak/label_stats_unif.csv")
        label_dist_df.to_csv(label_stats_filename)
        print("Label distribution statistics saved to:", label_stats_filename)

        # Save perturbed dataset
        perturbed_data_filename = os.path.join(output_dir, "data_perturbed/raw/data_train_perturbed_unif.csv")
        data_train.to_csv(perturbed_data_filename)
        print("Perturbed dataset saved to:", perturbed_data_filename)

    if args.perturb_function == 'perturb_label_stratified':
        label_stats_filename = os.path.join(output_dir, "descriptives/weak/label_stats_strat.csv")
        label_dist_df.to_csv(label_stats_filename)
        print("Label distribution statistics saved to:", label_stats_filename)

        # Save perturbed dataset
        perturbed_data_filename = os.path.join(output_dir, "data_perturbed/raw/data_train_perturbed_strat.csv")
        data_train.to_csv(perturbed_data_filename)
        print("Perturbed dataset saved to:", perturbed_data_filename)


if __name__ == '__main__':
    main()
