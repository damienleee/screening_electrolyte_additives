import deepchem as dc
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import tensorflow as tf
from scipy.stats import norm
import json
from MyGraphConvModel import MyGraphConvModel
import sys

np.random.seed(13)
tf.random.set_seed(13) # set random seed for reproducibility

arguments = sys.argv
model_dir = arguments[1]
curr_iteration = int(arguments[2])

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
        z = (curr_min - mean) / std
        return  (curr_min - mean)*norm.cdf(z) + std*norm.pdf(z)

def get_opt_score_and_EI(unlabelled_data_file, curr_min):
    df = pd.read_csv(unlabelled_data_file)
    loader = dc.data.CSVLoader([],\
               feature_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())
    to_predict = loader.create_dataset(unlabelled_data_file)
    model = MyGraphConvModel(n_tasks=4, graph_conv_layers=[545, 545], dense_layers=[562, 562, 562, 562], dropout=0.030050724302585315, \
learning_rate=0.0006959499230450821, uncertainty=True, model_dir=model_dir)
    model.restore()
    predictions, uncertainty = model.predict_uncertainty(to_predict)
    df[["Predicted LUMO", "Predicted Adsorption (100)", "Predicted Adsorption (110)", "Predicted Adsorption (111)"]] = predictions
    df[["Uncertainty in LUMO", "Uncertainty in Adsorption (100)", "Uncertainty in Adsorption (110)", "Uncertainty in Adsorption (111)"]] = uncertainty
    print(df)
    
    # set weights for multi-objective optimization
    w1 = 1/2
    w2 = w3 = w4 = 1/6
    predicted_LUMO = df["Predicted LUMO"]
    adsorption_100 = df["Predicted Adsorption (100)"]
    adsorption_110 = df["Predicted Adsorption (110)"]
    adsorption_111 = df["Predicted Adsorption (111)"]
    uncertainty_LUMO = df["Uncertainty in LUMO"]
    uncertainty_adsorption_100 = df["Uncertainty in Adsorption (100)"]
    uncertainty_adsorption_110 = df["Uncertainty in Adsorption (110)"]
    uncertainty_adsorption_111 = df["Uncertainty in Adsorption (111)"]
    predicted_opt_score = w1*-predicted_LUMO + w2*adsorption_100 + w3*adsorption_110 + w4*adsorption_111
    prediction_uncertainty = np.sqrt(w1**2*uncertainty_LUMO**2 + w2**2*uncertainty_adsorption_100**2 \
        + w3**2*uncertainty_adsorption_110**2 + w4**2*uncertainty_adsorption_111**2)
    
    df["predicted optimization score"] = predicted_opt_score
    df["prediction uncertainty"] = prediction_uncertainty
    
    df['EI score'] = df.apply(
        lambda row: EI_score(curr_min, row["predicted optimization score"], row["prediction uncertainty"]), axis=1)
    
    df.sort_values("EI score", inplace=True, ascending=False)
    df.to_csv("predictions_and_EI_scores.csv", index=False)
    
with open('BO_data.json', "r") as f:
    data = json.load(f)
curr_min = data[f"iteration_{curr_iteration-1}"]["Current best score"]
remove_failed_structures(f"iteration_{curr_iteration}.csv")
get_opt_score_and_EI(f"iteration_{curr_iteration}.csv", curr_min)