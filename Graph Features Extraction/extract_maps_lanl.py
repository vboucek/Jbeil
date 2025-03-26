import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

# Define file path (adjust as needed)
auth_path = None

assert auth_path, "Provide a path to the auth.txt file"

# Initialize nested dictionaries using defaultdict.
# Desired structure for each map: outer key -> inner key -> day -> count.
InHostUserMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
InHostSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
InHostUsrSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
OutHostUserMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
OutHostDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
OutHostUsrDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Define the chunk size (adjust based on memory constraints)
chunksize = 10000

# Create an iterator over chunks and wrap it in tqdm for progress display.
chunk_iter = pd.read_csv(
    auth_path,
    header=None,
    names=['time', 'src_t', 'user', 'src', 'dst', 'auth_t', 'log_t', 'auth_o', 's_f'],
    usecols=['time', 'user', 'src', 'dst'],
    chunksize=chunksize
)

# Process each chunk
for chunk in tqdm(chunk_iter, desc="Processing chunks"):
    # Compute the "day" column (integer division by 86400)
    chunk['day'] = (chunk['time'] // 86400).astype(int)

    # --- Inbound maps ---
    # InHostUserMap: group by dst, user, day
    gb = chunk.groupby(['dst', 'user', 'day']).size().reset_index(name='count')
    for row in gb.itertuples(index=False):
        InHostUserMap[row.dst][row.user][row.day] += row.count

    # InHostSrcMap: group by dst, src, day
    gb = chunk.groupby(['dst', 'src', 'day']).size().reset_index(name='count')
    for row in gb.itertuples(index=False):
        InHostSrcMap[row.dst][row.src][row.day] += row.count

    # InHostUsrSrcMap: group by dst, (user+src) and day
    chunk['user_src'] = chunk['user'] + chunk['src']
    gb = chunk.groupby(['dst', 'user_src', 'day']).size().reset_index(name='count')
    for row in gb.itertuples(index=False):
        InHostUsrSrcMap[row.dst][row.user_src][row.day] += row.count

    # --- Outbound maps ---
    # OutHostUserMap: group by src, user, day
    gb = chunk.groupby(['src', 'user', 'day']).size().reset_index(name='count')
    for row in gb.itertuples(index=False):
        OutHostUserMap[row.src][row.user][row.day] += row.count

    # OutHostDstMap: group by src, dst, day
    gb = chunk.groupby(['src', 'dst', 'day']).size().reset_index(name='count')
    for row in gb.itertuples(index=False):
        OutHostDstMap[row.src][row.dst][row.day] += row.count

    # OutHostUsrDstMap: group by src, (user+dst) and day
    chunk['user_dst'] = chunk['user'] + chunk['dst']
    gb = chunk.groupby(['src', 'user_dst', 'day']).size().reset_index(name='count')
    for row in gb.itertuples(index=False):
        OutHostUsrDstMap[row.src][row.user_dst][row.day] += row.count


# --- Convert defaultdicts to plain dictionaries recursively ---
def rec_dd_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: rec_dd_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: rec_dd_to_dict(v) for k, v in d.items()}
    return d


InHostUserMap = rec_dd_to_dict(InHostUserMap)
InHostSrcMap = rec_dd_to_dict(InHostSrcMap)
InHostUsrSrcMap = rec_dd_to_dict(InHostUsrSrcMap)
OutHostUserMap = rec_dd_to_dict(OutHostUserMap)
OutHostDstMap = rec_dd_to_dict(OutHostDstMap)
OutHostUsrDstMap = rec_dd_to_dict(OutHostUsrDstMap)

# --- Save maps to pickle files ---
with open("InHostUserMapLanl.pkl", "wb") as f:
    pickle.dump(InHostUserMap, f)
with open("InHostSrcMapLanl.pkl", "wb") as f:
    pickle.dump(InHostSrcMap, f)
with open("InHostUsrSrcMapLanl.pkl", "wb") as f:
    pickle.dump(InHostUsrSrcMap, f)
with open("OutHostUserMapLanl.pkl", "wb") as f:
    pickle.dump(OutHostUserMap, f)
with open("OutHostDstMapLanl.pkl", "wb") as f:
    pickle.dump(OutHostDstMap, f)
with open("OutHostUsrDstMapLanl.pkl", "wb") as f:
    pickle.dump(OutHostUsrDstMap, f)

print("All maps built and saved successfully.")
