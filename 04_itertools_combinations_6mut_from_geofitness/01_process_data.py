import itertools
import pandas as pd

amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
amino_acid_dict = {}
for index, value in enumerate(amino_acid_list):
    amino_acid_dict[value] = index

wt_seq = "MNKTIDQVRKGDRKSDLPVRRRPRRSAEETRRDILAKAEELFRERGFNAVAIADIASALNMSPANVFKHFSSKNALVDAIGFGQIGVFERQICPLDKSHAPLDRLRHLARNLMEQHHQDHFKHIRVFIQILMTAKQDMKCGDYYKSVIAKLLAEIIRDGVEAGLYIATDIPVLAETVLHALTSVIHPVLIAQEDIGNLATRCDQLVDLIDAGLRNPLAK"

fitness = pd.read_csv("../predicted_GeoFitness.csv", index_col=0)

tmp_dict = {214: ["N214A", "N214S"], 141: ["D141R", "D141L", "D141K", "D141S"], 124: ["R124D", "R124E", "R124S"], 77: ["D77L"], 50: ["A50K"], 97: ["S97K"], 99: ["A99K"], 196: ["N196K"], 199: ["T199K"]}

nosorted_result_list = []
for multi_mut_info in itertools.combinations(tmp_dict, 6):
    tmp_list = [tmp_dict[i] for i in multi_mut_info]
    nosorted_result_list += list(itertools.product(*tmp_list))

sorted_result_list = []
for i in nosorted_result_list:
    tmp = sorted(i, key=lambda x: int(x[1:-1]))
    sorted_result_list.append(",".join(tmp))

data = pd.DataFrame()
for multi_mut_info in sorted_result_list:
    data.loc[multi_mut_info, "wt_seq"] = wt_seq
    mut_seq = list(wt_seq)
    for single_mut_info in multi_mut_info.split(","):
        mut_seq[int(single_mut_info[1:-1])] = single_mut_info[-1]
    data.loc[multi_mut_info, "mut_seq"] = "".join(mut_seq)

data.index.name = "name"
data.to_csv("../data_pre-predict_by_GeoFitness.csv")
