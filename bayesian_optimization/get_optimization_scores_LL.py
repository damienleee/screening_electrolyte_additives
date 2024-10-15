import pandas as pd
import numpy as np
import json, sys

arguments = sys.argv
iteration_no = arguments[1]

LUMO_mean = 0.744633716
LUMO_std = 0.684535734
adsorption_100_mean = -2.153702316
adsorption_100_std = 1.173483951
adsorption_110_mean = -3.257926828
adsorption_110_std = 1.144809671
adsorption_111_mean = -3.28945536
adsorption_111_std = 1.088610496
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
std_scaling(df, "B3LYPLUMO", LUMO_mean, LUMO_std)
std_scaling(df, "Minimum Adsorption energy (100)", adsorption_100_mean, adsorption_100_std)
std_scaling(df, "Minimum Adsorption energy (110)", adsorption_110_mean, adsorption_110_std)
std_scaling(df, "Minimum Adsorption energy (111)", adsorption_111_mean, adsorption_111_std)
df['Optimization score'] = -w1*df["B3LYPLUMO"] + w2*df["Minimum Adsorption energy (100)"] + w3*df["Minimum Adsorption energy (110)"] + w4*df["Minimum Adsorption energy (111)"]
df.sort_values("Optimization score", inplace=True)

with open('BO_data.json', "r") as f:
   data = json.load(f)
   
# data[f"iteration_{iteration_no}"]["Current best score"] = df.iloc[0]['Optimization score']
# with open('BO_data.json', 'w') as f:
#    json.dump(data, f, indent=4)

df.to_csv("./data/labelled_data_scores.csv", index=False)