import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        # Wrap the file iterator with tqdm for a progress bar.
        for idx, line in tqdm(enumerate(f), desc="Processing lines", unit="line"):
            e = line.strip().split(',')
            u = int(e[1])
            i = int(e[2])
            ts = float(e[0])
            label = float(e[9])
            feat = np.array([float(e[10]), float(e[11]), float(e[12]),
                             float(e[13]), float(e[14]), float(e[15]),
                             float(e[16]), float(e[17]), float(e[18])
                             ])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = '/Users/vboucek/Desktop/Jbeil/dataset/lanl/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feats = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feats.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feats])
    max_idx = max(new_df.u.max(), new_df.i.max())

    rand_feat = np.eye(max_idx + 1)

    new_df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run("lanl", bipartite=args.bipartite)
