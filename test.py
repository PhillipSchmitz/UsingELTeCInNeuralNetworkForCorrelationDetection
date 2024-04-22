import pandas as pd
data = pd.read_csv("ELTeC-eng-dataset_2000tok-2000mfw.csv", sep=";")
print(data["a"].values)