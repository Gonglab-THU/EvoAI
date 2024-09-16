import os
import pandas as pd

data = pd.read_csv("../data_pre-predict_by_GeoFitness.csv", index_col=0)
wt_seq = list(set(data["wt_seq"]))[0]
wt_seq_list = list(wt_seq)

for name in data.index:
    os.system(f"mkdir -p ../mut_data/{name}")
    with open(f"../mut_data/{name}/result.fasta", "w") as f:
        f.write(">result\n{}".format(data.loc[name, "mut_seq"]))

    tmp = ""
    for single_mut_info in name.split(","):
        tmp += single_mut_info[0] + "A" + single_mut_info[1:] + ","
    with open(f"../mut_data/{name}/individual_list.txt", "w") as f:
        f.write(tmp[:-1] + ";")
