import pandas as pd
import numpy as np
import random
import os


def perturb_label_uniform(label, perturb_share, all_types):
    """
    Function to perturb labels uniformly, not taking into account pre-perturbation label distributions.

    :param label: The BIO label for an entity.
    :param perturb_share: The share of perturbed labels.
    :param all_types: The set of all occurring BIO label types for the dataset at hand.
    :return: The (perturbed) label.
    """
    if np.random.sample(1) < perturb_share:
        return random.sample(all_types, 1)[0]
    else:
        return label


def perturb_label_list(label_list, perturb_share, all_types, perturb_function=perturb_label_uniform):
    """
    Wrapper function for the perturbation functions, applies the function to a list of labels.
    """
    return [perturb_function(label, perturb_share, all_types) for label in label_list]


def main(data_dir):
    # Read the data
    data_train = pd.read_csv(os.path.join(data_dir, 'data_train.csv'), index_col=0, converters={'sequence_tok': pd.eval, 'ner_BIO_full': pd.eval})
    bio_labels = open(os.path.join(data_dir, 'COVIDNEWS_CONTROSTER/types.txt'), 'r').readlines()

    ALL_TYPES = set(" ".join(data_train["labels"].apply(lambda x: x.strip("\n"))).split(" "))

    perturb_shares = [0.05, 0.1, 0.15, 0.2, 0.25]  # List of perturbation shares

    # Create output directory in the "dataset" folder
    output_dir = os.path.join("dataset", "perturbation_output")
    os.makedirs(output_dir, exist_ok=True)

    for perturb_share in perturb_shares:
        # Apply perturbation to labels
        random.seed(123)  # Add seed for reproducibility
        data_train["ner_BIO_full" + "_" + str(perturb_share)] = data_train["ner_BIO_full"].apply(
            perturb_label_list, perturb_share=perturb_share, all_types=ALL_TYPES)

        # Generate statistics of label distributions
        label_stats = data_train["ner_BIO_full" + "_" + str(perturb_share)].apply(pd.Series.value_counts)

        # Print and save label distribution statistics
        label_stats_filename = os.path.join(output_dir, f"label_stats_{str(perturb_share)}.txt")
        with open(label_stats_filename, 'w') as file:
            print(label_stats, file=file)
        print("Label distribution statistics saved to:", label_stats_filename)

        # Save perturbed dataset
        perturbed_data_filename = os.path.join(output_dir, f"data_train_perturbed_{str(perturb_share)}.csv")
        data_train.to_csv(perturbed_data_filename)
        print("Perturbed dataset saved to:", perturbed_data_filename)


if __name__ == '__main__':
    data_input_dir = "../data/COVIDNEWS/"
    main(data_input_dir)
