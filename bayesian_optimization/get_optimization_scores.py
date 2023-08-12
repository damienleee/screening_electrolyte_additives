import pandas as pd
import numpy as np
import json, sys

arguments = sys.argv
iteration_no = arguments[1]

def get_optimization_score(row):
    # set weights for multi-objective optimization
    w1 = 1/2
    w2 = w3 = w4 = 1/6
    LUMO = -row["B3LYPLUMO"]
    adsorption_100 = row["Minimum Adsorption energy (100)"]
    adsorption_110 = row["Minimum Adsorption energy (110)"]
    adsorption_111 = row["Minimum Adsorption energy (111)"]
    return w1*LUMO + w2*adsorption_100 + w3*adsorption_110 + w4*adsorption_111

df = pd.read_csv("./labelled_data.csv")
df['Optimization score'] = df.apply(lambda row: get_optimization_score(row), axis=1)
df.sort_values("Optimization score", inplace=True)

with open('BO_data.json', "r") as f:
   data = json.load(f)
   
data[f"iteration_{iteration_no}"]["Current best score"] = df.iloc[0]['Optimization score']
with open('BO_data.json', 'w') as f:
   json.dump(data, f, indent=4)

df.to_csv("./labelled_data_scores.csv", index=False)