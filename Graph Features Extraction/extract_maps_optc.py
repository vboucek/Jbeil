import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

# Define file path for OPTC dataset.
optc_path = None
assert optc_path, "Provide a path to the OpTC file"

# Initialize nested dictionaries using defaultdict.
InHostSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
OutHostDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

chunksize = 10000

# Read the CSV in chunks (without parse_dates, then convert manually).
chunk_iter = pd.read_csv(
    optc_path,
    header=0,
    usecols=['timestamp', 'src_ip', 'dest_ip'],
    chunksize=chunksize
)

for chunk in tqdm(chunk_iter, desc="Processing chunks"):
    # Convert timestamp column to datetime (with utc=True)
    chunk.loc[:, 'timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True, errors='coerce')
    # Drop rows where conversion failed and explicitly copy.
    chunk = chunk.dropna(subset=['timestamp']).copy()

    # Convert datetime to Unix seconds by applying a lambda to extract the underlying nanosecond value.
    chunk.loc[:, 'timestamp_unix'] = chunk['timestamp'].apply(lambda x: x.value // 10 ** 9)
    chunk.loc[:, 'day'] = (chunk['timestamp_unix'] // 86400).astype(int)

    # Rename columns for consistency.
    chunk.rename(columns={'src_ip': 'src', 'dest_ip': 'dst'}, inplace=True)

    # --- Inbound Map ---
    # Group by destination (dst), source (src), and day.
    gb_in = chunk.groupby(['dst', 'src', 'day']).size().reset_index(name='count')
    for row in gb_in.itertuples(index=False):
        InHostSrcMap[row.dst][row.src][row.day] += row.count

    # --- Outbound Map ---
    # Group by source (src), destination (dst), and day.
    gb_out = chunk.groupby(['src', 'dst', 'day']).size().reset_index(name='count')
    for row in gb_out.itertuples(index=False):
        OutHostDstMap[row.src][row.dst][row.day] += row.count


def rec_dd_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: rec_dd_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        return {k: rec_dd_to_dict(v) for k, v in d.items()}
    else:
        return d


InHostSrcMap = rec_dd_to_dict(InHostSrcMap)
OutHostDstMap = rec_dd_to_dict(OutHostDstMap)

with open("InHostSrcMapOptc.pkl", "wb") as f:
    pickle.dump(InHostSrcMap, f)
with open("OutHostDstMapOptc.pkl", "wb") as f:
    pickle.dump(OutHostDstMap, f)

print("Maps built and saved successfully.")
