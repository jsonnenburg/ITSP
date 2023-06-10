import pandas as pd
import argparse
import os


def join_strings(row):
    return ' '.join(row)


def generate_labels(sequence, entity_dict):
    labels = []
    for word in sequence.split():
        if word in entity_dict:
            labels.append(entity_dict[word])
        else:
            labels.append('O')
    return labels


def main():
    parser = argparse.ArgumentParser()

    # data preparation parameters
    parser.add_argument("--dataset_dir",
                        default="COVIDNEWS",
                        type=str,
                        help="the input dataset directory.")

    args = parser.parse_args()

    # read in data
    data_train = pd.read_csv(os.path.join('../data', args.dataset_dir, '__init/weak/data_train.csv'), index_col=0)
    entity_dict = pd.read_csv(os.path.join('../data', args.dataset_dir, 'dist_skweak/entity_dict/skweak_dist.csv'),
                              header=None)

    output_dir = os.path.join('../data', args.dataset_dir, 'dist_skweak')
    os.makedirs(output_dir, exist_ok=True)

    # retrieve entities and labels from entity dict
    entity_dict = entity_dict[0]
    entity_dict.dropna(inplace=True)

    labels = entity_dict.apply(lambda x: x.split(' ')[-1])
    entities = entity_dict.apply(lambda x: x.rsplit(' ', 1)[0])
    entity_dict = dict(zip(entities, labels))

    data_train['labels_tok'] = data_train['sequence'].apply(lambda x: generate_labels(x, entity_dict))
    data_train['labels'] = data_train['labels_tok'].apply(lambda x: ' '.join(x))

    data_train.to_csv(os.path.join(output_dir, "data_train.csv"))
    print("Saved dataset to:", os.path.join(output_dir, "data_train.csv"))


if __name__ == '__main__':
    main()
