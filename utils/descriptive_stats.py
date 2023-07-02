import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_data(dataset):
    """
    dataset has to be in {COVIDNEWS, SciERC, WNUT16}.
    """
    data_train_weak = pd.read_csv(f'../data/{dataset}/__init/weak/data_train.csv', index_col=0)
    data_train_strong = pd.read_csv(f'../data/{dataset}/__init/strong/data_train.csv', index_col=0)
    data_test_strong = pd.read_csv(f'../data/{dataset}/__init/strong/data_test.csv', index_col=0)

    return data_train_weak, data_train_strong, data_test_strong


def plot_sequence_dist(dataset):
    tr_w, tr_s, ts_s = get_data(dataset)

    sl_tr_w = tr_w['sequence_tok'].apply(lambda x: len(x))
    sl_tr_s = tr_s['sequence_tok'].apply(lambda x: len(x))
    sl_ts_s = ts_s['sequence_tok'].apply(lambda x: len(x))

    sl_tr_w_mean = np.mean(sl_tr_w)
    sl_tr_s_mean = np.mean(sl_tr_s)
    sl_ts_s_mean = np.mean(sl_ts_s)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(sl_tr_w, shade=True, label='Weak Train')
    sns.kdeplot(sl_tr_s, shade=True, label='Strong Train')
    sns.kdeplot(sl_ts_s, shade=True, label='Strong Test')
    plt.axvline(sl_tr_w_mean, color='blue', label=f"Weak Train Mean ({sl_tr_w_mean.round(1)})")
    plt.axvline(sl_tr_s_mean, color='orange', label=f"Strong Train Mean ({sl_tr_s_mean.round(1)})")
    plt.axvline(sl_ts_s_mean, color='green', label=f"Strong Test Mean ({sl_ts_s_mean.round(1)})")
    plt.xlabel('Sequence Length')
    plt.legend()

    plt.savefig(f'../data/{dataset}/descriptives/{dataset}_sequence_lengths.pdf', bbox_inches='tight')
    print("Plot saved.")


def main():
    plot_sequence_dist('COVIDNEWS')
    plot_sequence_dist('SciERC')
    plot_sequence_dist('WNUT16')


if __name__ == "__main__":
    main()
