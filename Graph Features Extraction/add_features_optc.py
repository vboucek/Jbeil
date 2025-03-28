import os
import pickle
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Load mapping pickle files for feature calculations
# -----------------------------
with open('InHostSrcMapOptc.pkl', 'rb') as f:
    InHostSrcMap = pickle.load(f)
with open('OutHostDstMapOptc.pkl', 'rb') as f:
    OutHostDstMap = pickle.load(f)


# -----------------------------
# Helper functions for feature calculations
# -----------------------------
def average(lst):
    return sum(lst) / len(lst) if lst else 0


def stddev(lst):
    if not lst:
        return 0
    mean = average(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    return variance ** 0.5


# Inbound feature: for a given destination, count distinct sources.
def calculate_i_dsrc(dst):
    if dst in InHostSrcMap:
        return len(list(InHostSrcMap[dst].values()))
    else:
        return 0


# Outbound degree feature: for a given source, count distinct destinations.
def calculate_o_ddst(src):
    if src in OutHostDstMap:
        return len(list(OutHostDstMap[src].values()))
    else:
        return 0


# Outbound aggregated frequency feature: for a given source, compute the average frequency.
def calculate_oda_fdst(src):
    freq_list = []
    if src in OutHostDstMap:
        for dst in OutHostDstMap[src]:
            freq_list.append(average(list(OutHostDstMap[src][dst].values())))
        return average(freq_list) if freq_list else 0
    else:
        return 0


def save_map(m, fname, dst_path):
    m_rev = [None] * (max(m.values()) + 1)
    for k, v in m.items():
        m_rev[v] = k
    with open(os.path.join(dst_path, fname), "wb") as f:
        pickle.dump(m_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(os.path.join(dst_path, fname) + " saved")


def get_or_add(n, m, id):
    if n not in m:
        m[n] = id[0]
        id[0] += 1
    return m[n]


def split(optc_file: str, dst_path: str):
    """
    Processes the OPTC dataset and writes all records into one output file.
    Expected CSV file format (comma-separated):
      index,timestamp,src_ip,dest_ip,label
    The output CSV will include:
      ts, mapped_src, mapped_dest, label, calculate_i_dsrc(dest), calculate_o_ddst(src), calculate_oda_fdst(src)

    Note: The timestamp is a string and is converted into a Unix timestamp.
    """
    f_in = open(optc_file, "r")
    output_file = os.path.join(dst_path, "graph.csv")
    f_out = open(output_file, "w")

    # Skip header line of input file.
    header = f_in.readline()
    line = f_in.readline()

    # Mapping dictionary for src and dest.
    nmap = {}
    nid = [0]

    prog = tqdm(desc="Processing OPTC records")

    while line:
        tokens = line.strip().split(",")
        if len(tokens) < 5:
            line = f_in.readline()
            continue

        # Expected tokens:
        # [0]: index, [1]: timestamp, [2]: src_ip, [3]: dest_ip, [4]: label
        ts_str = tokens[1].strip()
        # Convert the timestamp string to a Unix timestamp.
        ts_unix = pd.to_datetime(ts_str).timestamp()
        src = tokens[2].strip()
        dest = tokens[3].strip()
        label = tokens[4].strip()

        mapped_src = get_or_add(src, nmap, nid)
        mapped_dest = get_or_add(dest, nmap, nid)

        calc_i = calculate_i_dsrc(dest)
        calc_o = calculate_o_ddst(src)
        calc_f = calculate_oda_fdst(src)

        out_line = "{},{},{},{},{},{},{}\n".format(ts_unix, mapped_src, mapped_dest, label, calc_i, calc_o, calc_f)
        f_out.write(out_line)

        prog.update(1)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(nmap, "nmap.pkl", dst_path)


def reverse_load_map(auth_path, fname):
    # Reverse a mapping pickle (list to dict)
    m = {}
    with open(os.path.join(auth_path, fname), "rb") as f:
        line_l = pickle.load(f)
        for i in range(len(line_l)):
            m[line_l[i]] = i
    return m


if __name__ == "__main__":
    optc_file = None
    assert optc_file, "Provide source path where OpTC file is located"
    dst_path = "../dataset/optc"
    os.makedirs(dst_path, exist_ok=True)
    split(optc_file, dst_path)
