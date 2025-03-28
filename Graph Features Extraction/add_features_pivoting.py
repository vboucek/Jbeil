import os
import pickle
from tqdm import tqdm

# -----------------------------
# Load mapping pickle files for feature calculations
# -----------------------------
with open('InHostSrcMapPivoting.pkl', 'rb') as f:
    InHostSrcMap = pickle.load(f)
with open('OutHostDstMapPivoting.pkl', 'rb') as f:
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


def split(pivoting_file: str, dst_path: str):
    """
    Processes the pivoting dataset and writes all records into one output file.
    Expected CSV file format (comma-separated):
      ,src,dst,p_src,p_dst,b_in,b_out,d,timestamp,pkts_in,pkts_out,proto,is_pivoting
    The output CSV will include:
      ts, mapped src, mapped dst, mapped p_src, mapped p_dst, mapped b_in, mapped b_out,
      d, pkts_in, pkts_out, proto, label,
      calculate_i_dsrc(dst), calculate_o_ddst(src), calculate_oda_fdst(src)
    """
    f_in = open(pivoting_file, "r")
    output_file = os.path.join(dst_path, "graph.csv")
    f_out = open(output_file, "w")

    # Skip header line and read the first data line.
    header = f_in.readline()
    line = f_in.readline()

    # Mapping dictionaries for nodes and additional fields.
    nmap = {}
    nid = [0]
    psmap = {}  # for p_src
    psid = [0]
    pdmap = {}  # for p_dst
    pdid = [0]
    bimap = {}  # for b_in
    bid = [0]
    bomap = {}  # for b_out
    bid2 = [0]

    prog = tqdm(desc="Processing records")

    # Format line lambda:
    # Expected tokens (after splitting on comma):
    # [0]: index, [1]: src, [2]: dst, [3]: p_src, [4]: p_dst, [5]: b_in,
    # [6]: b_out, [7]: d, [8]: timestamp, [9]: pkts_in, [10]: pkts_out,
    # [11]: proto, [12]: is_pivoting (label)
    fmt_line = lambda ts, src, dst, p_src, p_dst, b_in, b_out, d_field, pkts_in, pkts_out, proto, label: (
        "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            ts,
            get_or_add(src, nmap, nid),
            get_or_add(dst, nmap, nid),
            get_or_add(p_src, psmap, psid),
            get_or_add(p_dst, pdmap, pdid),
            get_or_add(b_in, bimap, bid),
            get_or_add(b_out, bomap, bid2),
            d_field,
            pkts_in,
            pkts_out,
            proto,
            label,
            calculate_i_dsrc(dst),
            calculate_o_ddst(src),
            calculate_oda_fdst(src)
        ),
        int(float(ts))
    )

    while line:
        tokens = line.strip().split(",")
        if len(tokens) < 13:
            line = f_in.readline()
            continue

        # Extract fields from the CSV (ignoring the index at tokens[0])
        src = tokens[1].strip()
        dst = tokens[2].strip()
        p_src = tokens[3].strip()
        p_dst = tokens[4].strip()
        b_in = tokens[5].strip()
        b_out = tokens[6].strip()
        d_field = tokens[7].strip()  # additional field "d"
        ts = tokens[8].strip()  # timestamp field
        pkts_in = tokens[9].strip()  # packets in
        pkts_out = tokens[10].strip()  # packets out
        proto = tokens[11].strip()  # protocol
        label = tokens[12].strip()  # is_pivoting (label)

        line_l, ts_val = fmt_line(ts, src, dst, p_src, p_dst, b_in, b_out, d_field, pkts_in, pkts_out, proto, label)
        f_out.write(line_l)
        prog.update(1)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(nmap, "nmap.pkl", dst_path)
    save_map(psmap, "psmap.pkl", dst_path)
    save_map(pdmap, "pdmap.pkl", dst_path)
    save_map(bimap, "bimap.pkl", dst_path)
    save_map(bomap, "bomap.pkl", dst_path)


def reverse_load_map(auth_path, fname):
    # Reverse a mapping pickle (list to dict)
    m = {}
    with open(os.path.join(auth_path, fname), "rb") as f:
        line_l = pickle.load(f)
        for i in range(len(line_l)):
            m[line_l[i]] = i
    return m


if __name__ == "__main__":
    pivoting_file = None
    assert pivoting_file, "Provide source path where Pivoting file is located"
    dst_path = "../dataset/pivoting"

    # Process the entire dataset into one file.
    split(pivoting_file, dst_path)
