import pickle

# Load data
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Check what's inside
print(f"Loaded {len(data)} map snapshots.")
print(f"Loaded {len(labels)} labels.")
