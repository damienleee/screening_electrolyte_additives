import pandas as pd
import numpy as np
import json, sys

arguments = sys.argv
iteration_no = arguments[1]

SEI_mean = -1.021141251
SEI_std = 0.737164506
adsorption_100_mean = -3.576637817
adsorption_100_std = 1.581444673
adsorption_110_mean = -4.69706615
adsorption_110_std = 1.54577794
adsorption_111_mean = -4.379219282
adsorption_111_std = 1.483376895
w1 = 1/2
w2 = w3 = w4 = 1/6

def std_scaling(df, column_name, mean, stddev, std=False):
    if not std:
        df[column_name] = (df[column_name] - mean)/stddev
    else:
        df[column_name] = (df[column_name])/stddev

def get_optimization_score(row):
    # set weights for multi-objective optimization
    
    LUMO = -row["B3LYPLUMO"]
    adsorption_100 = row["Minimum Adsorption energy (100)"]
    adsorption_110 = row["Minimum Adsorption energy (110)"]
    adsorption_111 = row["Minimum Adsorption energy (111)"]
    
    return w1*LUMO + w2*adsorption_100 + w3*adsorption_110 + w4*adsorption_111

df = pd.read_csv("./data/labelled_data.csv")
std_scaling(df, "B3LYPLUMO", SEI_mean, SEI_std)
std_scaling(df, "Minimum Adsorption energy (100)", adsorption_100_mean, adsorption_100_std)
std_scaling(df, "Minimum Adsorption energy (110)", adsorption_110_mean, adsorption_110_std)
std_scaling(df, "Minimum Adsorption energy (111)", adsorption_111_mean, adsorption_111_std)
df['Optimization score'] = w1*df["B3LYPLUMO"] + w2*df["Minimum Adsorption energy (100)"] + w3*df["Minimum Adsorption energy (110)"] + w4*df["Minimum Adsorption energy (111)"]
df.sort_values("Optimization score", inplace=True)

with open('BO_data.json', "r") as f:
   data = json.load(f)
   
#data[f"iteration_{iteration_no}"]["Current best score"] = df.iloc[0]['Optimization score']
#with open('BO_data.json', 'w') as f:
#   json.dump(data, f, indent=4)

df.to_csv("./labelled_data_scores.csv", index=False)