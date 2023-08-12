import deepchem as dc 
import pandas as pd
from MyGraphConvModel import MyGraphConvModel
import matplotlib.pyplot as plt

loader = dc.data.CSVLoader([],id_field="CID",
               feature_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())        
test_dataset = loader.create_dataset("pubchemqc_filtered_lumo.csv")

model = MyGraphConvModel(n_tasks=4, graph_conv_layers=[465, 465], dense_layers=[342, 342, 342], dropout=0.016595, \
learning_rate=0.0018738, uncertainty=True, model_dir="best_MAE_iteration_12")
model.restore()

embedding = model.predict_embedding(test_dataset)
df = pd.read_csv("pubchemqc_filtered_lumo.csv")
i = len(df)
#print(test_dataset.get_shape())
#print(embedding.shape)

embedding_df = pd.DataFrame(data=embedding[:i,:], index=df["CID"])
print(embedding[i:,:])
embedding_df.to_csv("embeddings.csv")