import os
import pickle
from tqdm import tqdm

# -----------------------------
# Load mapping pickle files for feature calculations
# -----------------------------
with open('InHostUserMap.pkl', 'rb') as f:
    InHostUserMap = pickle.load(f)
with open('InHostSrcMap.pkl', 'rb') as f:
    InHostSrcMap = pickle.load(f)
with open('InHostUsrSrcMap.pkl', 'rb') as f:
    InHostUsrSrcMap = pickle.load(f)
with open('OutHostDstMap.pkl', 'rb') as f:
    OutHostDstMap = pickle.load(f)
with open('OutHostUserMap.pkl', 'rb') as f:
    OutHostUserMap = pickle.load(f)
with open('OutHostUsrDstMap.pkl', 'rb') as f:
    OutHostUsrDstMap = pickle.load(f)


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


# Inbound feature functions (based on destination)
def calculate_i_dusr(dst):
    if dst in InHostUserMap:
        return len(list(InHostUserMap[dst].values()))
    else:
        return 0


def calculate_i_dsrc(dst):
    if dst in InHostSrcMap:
        return len(list(InHostSrcMap[dst].values()))
    else:
        return 0


def calculate_i_dusr_src(dst):
    if dst in InHostUsrSrcMap:
        return len(list(InHostUsrSrcMap[dst].values()))
    else:
        return 0


# Outbound degree feature functions (based on source)
def calculate_o_dusr(src):
    if src in OutHostUserMap:
        return len(list(OutHostUserMap[src].values()))
    else:
        return 0


def calculate_o_ddst(src):
    if src in OutHostDstMap:
        return len(list(OutHostDstMap[src].values()))
    else:
        return 0


def calculate_o_dusr_dst(src):
    if src in OutHostUsrDstMap:
        return len(list(OutHostUsrDstMap[src].values()))
    else:
        return 0


# Outbound aggregated frequency feature functions (based on source)
def calculate_oda_fusr(src):
    freq_list = []
    if src in OutHostUserMap:
        for user in OutHostUserMap[src]:
            freq_list.append(average(list(OutHostUserMap[src][user].values())))
        return average(freq_list) if freq_list else 0
    else:
        return 0


def calculate_oda_fdst(src):
    freq_list = []
    if src in OutHostDstMap:
        for dst in OutHostDstMap[src]:
            freq_list.append(average(list(OutHostDstMap[src][dst].values())))
        return average(freq_list) if freq_list else 0
    else:
        return 0


def calculate_oda_fusr_dst(src):
    freq_list = []
    if src in OutHostUsrDstMap:
        for usrdst in OutHostUsrDstMap[src]:
            freq_list.append(average(list(OutHostUsrDstMap[src][usrdst].values())))
        return average(freq_list) if freq_list else 0
    else:
        return 0


DELTA = 60


def mark_anoms(redteam_file: str):
    with open(redteam_file, "r") as f:
        red_events = f.read().split()
    red_events = red_events[1:]

    def add_ts(d, val, ts):
        val = (val[1], val[2])
        if val in d:
            d[val].append(ts)
        else:
            d[val] = [ts]

    anom_dict = {}
    for event in red_events:
        tokens = event.split(",")
        ts = int(tokens.pop(0))
        add_ts(anom_dict, tokens, ts)
    return anom_dict


def is_anomalous(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False
    times = d[(src, dst)]
    for time in times:
        if ts == time:
            return True
    return False


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


def split(auth_file: str, redteam_file: str, dst_path: str, duration_per_file: int = DELTA):
    # Load anomalies from redteam file (for fmt_label)
    anom_dict = mark_anoms(redteam_file)

    last_time = 1
    cur_time = 0

    f_in = open(auth_file, "r")
    f_out = open(os.path.join(dst_path, f"graph_{cur_time}.csv"), "w")

    # Skip header line and read first data line
    _ = f_in.readline()
    line = f_in.readline()

    nmap = {}
    nid = [0]
    umap = {}
    uid = [0]
    atmap = {}
    atid = [0]
    ltmap = {}
    ltid = [0]
    aomap = {}
    aoid = [0]
    smap = {}
    sid = [0]
    prog = tqdm(desc="Seconds parsed", total=5011199)

    # Formatting functions for source user extraction and label
    fmt_src = lambda x: x.split("@")[0].replace("$", "")
    fmt_label = lambda ts, src, dst: 1 if is_anomalous(anom_dict, src, dst, ts) else 0

    # Modified format line lambda:
    # Original 10 fields + 9 extra feature fields are appended.
    # Fields from the auth file: ts, src_user, dst_user, src_computer, dst_computer, auth_type, logon_type, auth_orientation, success
    # Mapping: tokens[0]=ts, [1]=src_user, [2]=dst_user, [3]=src, [4]=dst, [5]=auth_type, [6]=logon_type, [7]=auth_orientation, [8]=success
    fmt_line = lambda ts, src, dst, src_u, dst_u, auth_t, logon_t, auth_o, success: (
        "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            ts,
            get_or_add(src, nmap, nid),
            get_or_add(dst, nmap, nid),
            get_or_add(fmt_src(src_u), umap, uid),
            get_or_add(fmt_src(dst_u), umap, uid),
            get_or_add(auth_t, atmap, atid),
            get_or_add(logon_t, ltmap, ltid),
            get_or_add(auth_o, aomap, aoid),
            get_or_add(success, smap, sid),
            fmt_label(int(ts), src, dst),
            # Nine extra features:
            calculate_i_dusr(dst),
            calculate_i_dsrc(dst),
            calculate_i_dusr_src(dst),
            calculate_o_dusr(src),
            calculate_o_ddst(src),
            calculate_o_dusr_dst(src),
            calculate_oda_fusr(src),
            calculate_oda_fdst(src),
            calculate_oda_fusr_dst(src)
        ),
        int(ts)
    )

    while line:
        # Filtering for NTLM lines only
        if "NTLM" not in line.upper():
            line = f_in.readline()
            continue

        tokens = line.split(",")
        # tokens:
        # 0: ts, 1: src_user, 2: dst_user, 3: src, 4: dst, 5: auth_type, 6: logon_type, 7: auth_orientation, 8: success
        line_l, ts = fmt_line(
            tokens[0],
            tokens[3],
            tokens[4],
            tokens[1],
            tokens[2],
            tokens[5],
            tokens[6],
            tokens[7],
            tokens[8].strip(),
        )

        if ts != last_time:
            prog.update(ts - last_time)
            last_time = ts

        # After ts progresses at least duration_per_file, make a new file
        if ts >= cur_time + duration_per_file:
            f_out.close()
            cur_time += duration_per_file
            dst_file = os.path.join(dst_path, f"graph_{cur_time // duration_per_file}.csv")
            f_out = open(dst_file, "w+")

        f_out.write(line_l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(nmap, "nmap.pkl", dst_path)
    save_map(umap, "umap.pkl", dst_path)
    save_map(atmap, "atmap.pkl", dst_path)
    save_map(ltmap, "ltmap.pkl", dst_path)
    save_map(aomap, "aomap.pkl", dst_path)
    save_map(smap, "smap.pkl", dst_path)


def reverse_load_map(auth_path, fname):
    # mapping pickle is a list, need to reverse it to a dict
    m = {}
    with open(os.path.join(auth_path, fname), "rb") as f:
        line_l = pickle.load(f)
        for i in range(0, len(line_l)):
            m[line_l[i]] = i
    return m


if __name__ == "__main__":
    src_path = "/Volumes/KINGSTON/"
    dst_path = "../dataset/lanl"
    auth_file = os.path.join(src_path, "auth.txt")
    red_file = os.path.join(src_path, "redteam.txt")

    # Adjust duration_per_file as needed
    split(auth_file, red_file, dst_path, duration_per_file=1_000_000_000)
