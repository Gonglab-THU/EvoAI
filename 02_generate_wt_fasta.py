import os
import pandas as pd

data = pd.read_csv("data.csv", index_col=0)
wt_seq = list(set(data["wt_seq"]))[0]

os.system("mkdir -p wt_data")
with open("wt_data/result.fasta", "w") as f:
    f.write(f">result\n{wt_seq}")
