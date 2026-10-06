import csv

features_path = "data/raw/elliptic/txs_features.csv"

with open(features_path, "r", encoding="utf-8", newline="") as f:
    header = next(csv.reader(f))

print("\n========== COLUMN SUMMARY ==========")
print("Total columns:", len(header))

print("\n========== ALL COLUMN NAMES ==========")

for i, column in enumerate(header):
    print(f"{i}: {column}")