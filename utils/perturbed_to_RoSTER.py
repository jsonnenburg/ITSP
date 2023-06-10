import pandas as pd
import argparse
import os
import ast


def join_strings(row):
    return ' '.join(row)


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
    data_train_strong = pd.read_csv(os.path.join('../data', args.dataset_dir, '__init/strong/data_train.csv'),
                                    index_col=0)
    data_test_strong = pd.read_csv(os.path.join('../data', args.dataset_dir, '__init/strong/data_test.csv'), index_col=0)
    # read in the weakly annotated, perturbed training data
    data_train_weak = pd.read_csv(os.path.join('../data', args.dataset_dir,
                                               f'data_perturbed/BIO/data_train_perturbed_{args.sampling_method}.csv'),
                                  index_col=0)
    types = open(os.path.join('../data', args.dataset_dir, 'types.txt'), 'r').readlines()

    output_dir = os.path.join('../data', args.dataset_dir, 'data_RoSTER/')
    os.makedirs(output_dir, exist_ok=True)

    relev_cols = [col for col in data_train_weak if col.startswith('labels_tok')]

    ### CREATE FOLDER STRUCTURE FOR WEAK BIO LABELS
    for column in relev_cols:
        data_train_weak[column] = data_train_weak[column].astype(str)
        data_train_weak[column] = data_train_weak[column].apply(lambda x: ast.literal_eval(x))

        col_temp = data_train_weak[column].apply(join_strings)
        # create subdirectory
        if column == 'labels_tok':
            output_subdir = os.path.join(output_dir, args.sampling_method, 'baseline')
        else:
            output_subdir = os.path.join(output_dir, args.sampling_method, f'{column.split("_")[-1]}')
        os.makedirs(output_subdir, exist_ok=True)

        os.makedirs(os.path.join(output_subdir, 'strong'), exist_ok=True)
        os.makedirs(os.path.join(output_subdir, 'weak'), exist_ok=True)

        ## save weak data
        file_path = os.path.join(output_subdir, "weak/train_label_dist.txt")
        col_temp.to_csv(file_path, index=False, header=False)
        print("Saved distant training labels to:", file_path)

        # save sequences
        file_path = os.path.join(output_subdir, "weak/train_text.txt")
        data_train_weak['sequence'].to_csv(file_path, index=False, header=False)
        print("Saved weakly labeled sequences to:", file_path)

        # save types.txt to subfolder
        with open(os.path.join(output_subdir, 'weak/types.txt'), 'w') as file:
            # Write each word to a new line in the text file
            for word in types:
                file.write(word.strip() + '\n')

        # save strong data
        data_train_strong['labels_tok'] = data_train_strong['labels_tok'].astype(str)
        data_train_strong['labels_tok'] = data_train_strong['labels_tok'].apply(lambda x: ast.literal_eval(x))

        col_temp = data_train_strong['labels_tok'].apply(join_strings)
        file_path = os.path.join(output_subdir, f"strong/train_label_true.txt")
        col_temp.to_csv(file_path, index=False, header=False)
        print("Saved strong training labels to:", file_path)

        # save sequence
        file_path = os.path.join(output_subdir, f"strong/train_text.txt")
        data_train_strong['sequence'].to_csv(file_path, index=False, header=False)
        print("Saved strongly labeled training sequences to:", file_path)

        ### save sequences and labels - STRONG BIO LABELS (TEST SET)
        data_test_strong['labels_tok'] = data_test_strong['labels_tok'].astype(str)
        data_test_strong['labels_tok'] = data_test_strong['labels_tok'].apply(lambda x: ast.literal_eval(x))

        col_temp = data_test_strong['labels_tok'].apply(join_strings)
        file_path = os.path.join(output_subdir, f"strong/test_label_true.txt")
        col_temp.to_csv(file_path, index=False, header=False)
        print("Saved strong test labels to:", file_path)

        # save sequence
        file_path = os.path.join(output_subdir, f"strong/test_text.txt")
        data_test_strong['sequence'].to_csv(file_path, index=False, header=False)
        print("Saved strongly labeled test sequences to:", file_path)

        with open(os.path.join(output_subdir, 'strong/types.txt'), 'w') as file:
            # Write each word to a new line in the text file
            for word in types:
                file.write(word.strip() + '\n')


if __name__ == '__main__':
    main()
