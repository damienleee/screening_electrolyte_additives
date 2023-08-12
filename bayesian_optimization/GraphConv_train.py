import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import deepchem as dc
import tensorflow as tf
import numpy as np
from MyGraphConvModel import MyGraphConvModel
import csv
import sys

arguments = sys.argv
model_dir = arguments[1]

tf.keras.utils.set_random_seed(13)  # set random seed for reproducibility

"""
Training of GraphConvModel, early stopping is implemented manually.
"""

test_csv = "test_data.csv"
validation_csv = "validation_data.csv"
train_csv = "train_data.csv"
loader = dc.data.CSVLoader(["B3LYPLUMO", "Minimum Adsorption energy (100)", "Minimum Adsorption energy (110)", "Minimum Adsorption energy (111)"],\
               feature_field="SMILES", featurizer=dc.feat.ConvMolFeaturizer())
test_dataset = loader.create_dataset(test_csv)
validation_dataset = loader.create_dataset(validation_csv)
train_dataset = loader.create_dataset(train_csv)

model = MyGraphConvModel(n_tasks=4, graph_conv_layers=[545, 545], dense_layers=[562, 562, 562, 562], dropout=0.030050724302585315, \
learning_rate=0.0006959499230450821, uncertainty=True, model_dir=model_dir)
model.restore()

mae_metric = dc.metrics.Metric(dc.metrics.mae_score)
mse_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
r2_metric = dc.metrics.Metric(dc.metrics.r2_score)
metric_list = [mae_metric, mse_metric, r2_metric]

log = []
best_MAE = 999

for epoch in range(500):
    model.fit(train_dataset, nb_epoch=1)
    
    mean_train_score = model.evaluate(train_dataset, metric_list, [], per_task_metrics = False)
    mean_val_score = model.evaluate(validation_dataset, metric_list, [], per_task_metrics = False)
    MAE_val = mean_val_score["mae_score"]
    log.append([epoch+1, mean_train_score["mae_score"], mean_train_score["mean_squared_error"], MAE_val, mean_val_score["mean_squared_error"]])
    if MAE_val < best_MAE:
        model.save_checkpoint(model_dir=f"GraphConvModel/epoch_{epoch+1}_MAE_{MAE_val}")
        best_MAE = MAE_val
    
header = ["Epoch", "Training MAE", "Training loss", "Validation MAE", "Validation Loss"]

with open('history.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(log)