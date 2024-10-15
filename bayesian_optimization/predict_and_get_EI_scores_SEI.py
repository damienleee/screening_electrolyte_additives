import deepchem as dc
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import tensorflow as tf
from scipy.stats import norm
import json, sys
from MyGraphConvModel import MyGraphConvModel

np.random.seed(13)
tf.random.set_seed(13) # set random seed for reproducibility

SEI_mean = -1.021141251
SEI_std = 0.737164506
adsorption_100_mean = -3.576637817
adsorption_100_std = 1.581444673
adsorption_110_mean = -4.69706615
adsorption_110_std = 1.54577794
adsorption_111_mean = -4.379219282
adsorption_111_std = 1.483376895

arguments = sys.argv
model_dir = arguments[1]
curr_iteration = int(arguments[2])

# set weights for multi-objective optimization
w1 = 1/2
w2 = w3 = w4 = 1/6

def std_scaling(df, column_name, mean, stddev, std=False):
    if not std:
        df[column_name] = (df[column_name] - mean)/stddev
    else:
        df[column_name] = (df[column_name])/stddev

def remove_failed_structures(unlabelled_data_file):
    df = pd.read_csv(unlabelled_data_file)
    remove_indexes = []
    featurizer=dc.feat.ConvMolFeaturizer()
    for i, row in df.iterrows():
        features = featurizer.featurize(row["smiles"])
        if features.size == 0:
            remove_indexes.append(i)
    df = df.drop(remove_indexes)
    df.to_csv(unlabelled_data_file, index=False)
        
    
def EI_score(curr_min, mean, std):
    if std == 0:
        return 0
    else:
        z = (curr_min - mean - 0.01) / std
        return  (curr_min - mean)*norm.cdf(z) + std*norm.pdf(z)

def get_opt_score_and_EI(unlabelled_data_file, model_dir, curr_min):
    df = pd.read_csv(unlabelled_data_file)
    loader = dc.data.CSVLoader([],\
               feature_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())
    to_predict = loader.create_dataset(unlabelled_data_file)
    model = MyGraphConvModel(n_tasks=4, graph_conv_layers=[465, 465], dense_layers=[342, 342, 342], dropout=0.016595, \
learning_rate=0.0018738, uncertainty=True, model_dir=model_dir)
    model.restore()
    predictions, uncertainty = model.predict_uncertainty(to_predict)
    df[["Predicted LUMO", "Predicted Adsorption (100)", "Predicted Adsorption (110)", "Predicted Adsorption (111)"]] = predictions
    df[["Uncertainty in LUMO", "Uncertainty in Adsorption (100)", "Uncertainty in Adsorption (110)", "Uncertainty in Adsorption (111)"]] = uncertainty
    print(df)
    
    std_scaling(df, "Predicted LUMO", SEI_mean, SEI_std)
    std_scaling(df, "Predicted Adsorption (100)", adsorption_100_mean, adsorption_100_std)
    std_scaling(df, "Predicted Adsorption (110)", adsorption_110_mean, adsorption_110_std)
    std_scaling(df, "Predicted Adsorption (111)", adsorption_111_mean, adsorption_111_std)
    std_scaling(df, "Uncertainty in LUMO", SEI_mean, SEI_std, std=True)
    std_scaling(df, "Uncertainty in Adsorption (100)", adsorption_100_mean, adsorption_100_std, std=True)
    std_scaling(df, "Uncertainty in Adsorption (110)", adsorption_110_mean, adsorption_110_std, std=True)
    std_scaling(df, "Uncertainty in Adsorption (111)", adsorption_111_mean, adsorption_111_std, std=True)
    predicted_LUMO = df["Predicted LUMO"]
    adsorption_100 = df["Predicted Adsorption (100)"]
    adsorption_110 = df["Predicted Adsorption (110)"]
    adsorption_111 = df["Predicted Adsorption (111)"]
    uncertainty_LUMO = df["Uncertainty in LUMO"]
    uncertainty_adsorption_100 = df["Uncertainty in Adsorption (100)"]
    uncertainty_adsorption_110 = df["Uncertainty in Adsorption (110)"]
    uncertainty_adsorption_111 = df["Uncertainty in Adsorption (111)"]
    predicted_opt_score = w1*predicted_LUMO + w2*adsorption_100 + w3*adsorption_110 + w4*adsorption_111
    prediction_uncertainty = np.sqrt(w1**2*uncertainty_LUMO**2 + w2**2*uncertainty_adsorption_100**2 \
        + w3**2*uncertainty_adsorption_110**2 + w4**2*uncertainty_adsorption_111**2)
    
    df["predicted optimization score"] = predicted_opt_score
    df["prediction uncertainty"] = prediction_uncertainty
    
    df['EI score'] = df.apply(
        lambda row: EI_score(curr_min, row["predicted optimization score"], row["prediction uncertainty"]), axis=1)
    
    df.sort_values("EI score", inplace=True, ascending=False)
    df.to_csv("predictions_and_EI_scores.csv", index=False)
    
f = open('BO_data.json', "r")
data = json.load(f)
curr_min = data[f"iteration_{curr_iteration-1}"]["Current best score"]
#remove_failed_structures("iteration_1.csv")
get_opt_score_and_EI(f"iteration_18.csv", model_dir, curr_min)