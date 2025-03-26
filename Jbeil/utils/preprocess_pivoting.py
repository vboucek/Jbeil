import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import os


def preprocess(input_file):
    """
    Reads the previously generated graph.csv file line by line and builds:
      - a DataFrame with columns:
            'u': mapped src (from column 1)
            'i': mapped dst (from column 2)
            'ts': timestamp (from column 0)
            'label': label (from column 11)
            'idx': line number (starting at 0)
      - a NumPy array with features from the calculated columns:
            [calc_i_dsrc, calc_o_ddst, calc_oda_fdst] (from columns 12, 13, 14)
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_list = []
    idx_list = []

    with open(input_file, "r") as f:
        header = next(f)  # assume header exists; if not, remove this line
        for idx, line in tqdm(enumerate(f), desc="Processing lines", unit="line"):
            e = line.strip().split(',')
            if len(e) < 15:
                continue  # skip incomplete lines
            # Columns: 0: ts, 1: mapped src, 2: mapped dst, 3: p_src, 4: p_dst,
            # 5: b_in, 6: b_out, 7: d, 8: pkts_in, 9: pkts_out, 10: proto,
            # 11: label, 12: calc_i_dsrc, 13: calc_o_ddst, 14: calc_oda_fdst

            ts = float(e[0].strip())
            u = int(e[1].strip())
            i = int(e[2].strip())
            label = float(e[11].strip())
            # Use the calculated features (columns 12, 13, 14)
            f1 = float(e[12].strip())
            f2 = float(e[13].strip())
            f3 = float(e[14].strip())
            feat = np.array([f1, f2, f3])

            ts_list.append(ts)
            u_list.append(u)
            i_list.append(i)
            label_list.append(label)
            idx_list.append(idx)
            feat_list.append(feat)

    df = pd.DataFrame({
        'u': u_list,
        'i': i_list,
        'ts': ts_list,
        'label': label_list,
        'idx': idx_list
    })
    return df, np.array(feat_list)


def reindex(df, bipartite=True):
    """
    Reindexes the nodes. For a bipartite graph, nodes in the 'u' and 'i'
    sets are shifted such that they do not overlap.
    """
    new_df = df.copy()
    if bipartite:
        # Assert the node numbering is contiguous.
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
    # Create output directory
    Path("data/").mkdir(parents=True, exist_ok=True)
    # Input file: use the previously generated graph.csv.
    # Adjust the path as needed.
    INPUT_FILE = os.path.join("../../dataset/pivoting", "graph.csv")
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    # Preprocess the CSV file.
    df, feats = preprocess(INPUT_FILE)
    new_df = reindex(df, bipartite)

    # For consistency with prior code, add a zero-feature for index 0.
    empty = np.zeros(feats.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feats])

    max_idx = max(new_df.u.max(), new_df.i.max())
    # Create node features (identity matrix) for nodes.
    node_feat = np.eye(max_idx + 1)

    # Save the processed DataFrame and features.
    new_df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, node_feat)
    print("Preprocessing completed. Files saved to:", OUT_DF, OUT_FEAT, OUT_NODE_FEAT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
    args = parser.parse_args()
    run("pivoting", bipartite=args.bipartite)
