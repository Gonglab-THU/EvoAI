import pandas as pd

wt_seq = "MNKTIDQVRKGDRKSDLPVRRRPRRSAEETRRDILAKAEELFRERGFNAVAIADIASALNMSPANVFKHFSSKNALVDAIGFGQIGVFERQICPLDKSHAPLDRLRHLARNLMEQHHQDHFKHIRVFIQILMTAKQDMKCGDYYKSVIAKLLAEIIRDGVEAGLYIATDIPVLAETVLHALTSVIHPVLIAQEDIGNLATRCDQLVDLIDAGLRNPLAK"
data = pd.read_excel("fitness.xlsx", nrows=83)
data = data.drop(0)


def get_mut_info(multi_mut_info):
    tmp = ""
    for single_mut_info in multi_mut_info.strip().split(" "):
        assert wt_seq[int(single_mut_info[1:-1]) - 1] == single_mut_info[0]
        tmp += single_mut_info[0] + str(int(single_mut_info[1:-1]) - 1) + single_mut_info[-1] + ","
    return tmp[:-1]


data["name"] = data.apply(lambda x: get_mut_info(x["NONE"]), axis=1)

data = data.set_index("name")
data["wt_seq"] = wt_seq
data = data[["wt_seq", "FINAL SEQUENCE", "normalized fold change"]]
data.columns = ["wt_seq", "mut_seq", "score"]

data["score"] = data["score"] - 3.454773869
data.to_csv("data.csv")
