import pandas as pd

# generate csv
data = pd.read_csv("../data_pre-predict_by_GeoFitness.csv", index_col=0)
for name in data.index:
    with open(f"../mut_data/{name}/result_expression.txt", "r") as f:
        data.loc[name, "expression"] = float(f.read())

data = data.sort_values("expression", ascending=False)
data = data[["expression"]]
data.to_csv("sorted_predicted_result.csv")

# generate excel (mut_pos + 1)
for name in data.index:
    tmp = ""
    for i in name.split(","):
        tmp += i[0] + str(int(i[1:-1]) + 1) + i[-1] + ","
        data.loc[name, "mut_name"] = tmp[:-1]
data = data.set_index("mut_name")
data.to_csv("sorted_predicted_result.txt", sep="\t")
