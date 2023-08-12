import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import deepchem as dc
import tensorflow as tf
import numpy as np
from MyGraphConvModel import MyGraphConvModel
import json
import sys

arguments = sys.argv
model_dir = arguments[1]
iteration_no = arguments[2]

tf.keras.utils.set_random_seed(13)  # set random seed for reproducibility

"""
Testing of GraphConvModel.
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

f = open('BO_data.json', "r")
data = json.load(f)

mae_metric = dc.metrics.Metric(dc.metrics.mae_score)
mse_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
r2_metric = dc.metrics.Metric(dc.metrics.r2_score)
metric_list = [mae_metric, mse_metric, r2_metric]

to_store = {}
train_score = model.evaluate(train_dataset, metric_list, [], per_task_metrics = True)[1]
print("GraphConv Model per task train score: ",train_score)
mean_train_score = model.evaluate(train_dataset, metric_list, [], per_task_metrics = False)
print("GraphConv Model train score: ", mean_train_score)
to_store["training score per task"] = train_score
to_store["overall training score"] = mean_train_score

validation_score = model.evaluate(validation_dataset, metric_list, [], per_task_metrics = True)[1]
print("GraphConv Model per task validation score: ",validation_score)
mean_val_score = model.evaluate(validation_dataset, metric_list, [], per_task_metrics = False)
print("GraphConv Model validation score: ", mean_val_score)
to_store["validation score per task"] = validation_score
to_store["overall validation score"] = mean_val_score

test_score = model.evaluate(test_dataset, metric_list, [], per_task_metrics = True)[1]
print("GraphConv Model per task test score: ",test_score)
mean_test_score = model.evaluate(test_dataset, metric_list, [], per_task_metrics = False)
print("GraphConv Model test score: ", mean_test_score)
to_store["test score per task"] = test_score
to_store["overall test score"] = mean_test_score

data[f"iteration_{iteration_no}"] = to_store
with open('BO_data.json', "w") as f:
    json.dump(data, f)