import itertools
import pandas as pd

amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
amino_acid_dict = {}
for index, value in enumerate(amino_acid_list):
    amino_acid_dict[value] = index

wt_seq = "MNKTIDQVRKGDRKSDLPVRRRPRRSAEETRRDILAKAEELFRERGFNAVAIADIASALNMSPANVFKHFSSKNALVDAIGFGQIGVFERQICPLDKSHAPLDRLRHLARNLMEQHHQDHFKHIRVFIQILMTAKQDMKCGDYYKSVIAKLLAEIIRDGVEAGLYIATDIPVLAETVLHALTSVIHPVLIAQEDIGNLATRCDQLVDLIDAGLRNPLAK"

data = pd.read_csv("../data.csv", index_col=0)
tmp = []
for i in data.index.str.split(","):
    tmp += i

candidate_list = list(set(tmp))
train_tmp_dict = {}
for index, value in enumerate(candidate_list):
    mut_pos = int(value[1:-1])
    if mut_pos not in train_tmp_dict.keys():
        train_tmp_dict[mut_pos] = [value]
    else:
        train_tmp_dict[mut_pos].append(value)

fitness = pd.read_csv("../predicted_GeoFitness.csv", index_col=0).T

fitness_tmp_dict = {}
for mut_pos in train_tmp_dict.keys():
    value = 0
    for i in train_tmp_dict[mut_pos]:
        value += fitness.iloc[int(i[1:-1]), amino_acid_dict[i[-1]]]
    fitness_tmp_dict[mut_pos] = value / len(train_tmp_dict[mut_pos])

tmp_dict = {}
for mut_pos in sorted(fitness_tmp_dict, key=lambda x: fitness_tmp_dict[x], reverse=True)[:14]:
    if mut_pos >= 30:
        tmp_dict[mut_pos] = train_tmp_dict[mut_pos]

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
data.to_csv("../data_pre-predict_by_80train.csv")
