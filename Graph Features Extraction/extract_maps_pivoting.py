import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

# Define file path (adjust as needed)
pivoting_path = "/Volumes/KINGSTON/dataset_pivoting.csv"

assert pivoting_path, "Provide a path to the dataset_pivoting.csv file"

# Initialize nested dictionaries using defaultdict.
# Desired structure for each map: outer key -> inner key -> day -> count.
InHostSrcMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
OutHostDstMap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Define the chunk size (adjust based on memory constraints)
chunksize = 10000

# Read the CSV in chunks. Here we use 'usecols' to extract only the relevant columns.
# The dataset has a header row so we let pandas parse it accordingly.
chunk_iter = pd.read_csv(
    pivoting_path,
    header=0,  # dataset has a header row
    usecols=['timestamp', 'src', 'dst'],
    chunksize=chunksize
)

# Process each chunk
for chunk in tqdm(chunk_iter, desc="Processing chunks"):
    # Compute the "day" column (integer division by 86400).
    # This converts the timestamp into a day index.
    chunk['day'] = (chunk['timestamp'] // 86400).astype(int)

    # --- Inbound Map ---
    # InHostSrcMap: group by destination (dst), source (src), and day.
    gb_in = chunk.groupby(['dst', 'src', 'day']).size().reset_index(name='count')
    for row in gb_in.itertuples(index=False):
        InHostSrcMap[row.dst][row.src][row.day] += row.count

    # --- Outbound Map ---
    # OutHostDstMap: group by source (src), destination (dst), and day.
    gb_out = chunk.groupby(['src', 'dst', 'day']).size().reset_index(name='count')
    for row in gb_out.itertuples(index=False):
        OutHostDstMap[row.src][row.dst][row.day] += row.count


def rec_dd_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: rec_dd_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: rec_dd_to_dict(v) for k, v in d.items()}
    return d


InHostSrcMap = rec_dd_to_dict(InHostSrcMap)
OutHostDstMap = rec_dd_to_dict(OutHostDstMap)

# --- Save maps to pickle files ---
with open("InHostSrcMapPivoting.pkl", "wb") as f:
    pickle.dump(InHostSrcMap, f)
with open("OutHostDstMapPivoting.pkl", "wb") as f:
    pickle.dump(OutHostDstMap, f)

print("Maps built and saved successfully.")
