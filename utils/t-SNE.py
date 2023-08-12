import pandas as pd
from sklearn.manifold import TSNE

perplexity=100

df = pd.read_csv("embeddings.csv.gz",index_col="CID")
tsne = TSNE(n_components=2,perplexity=perplexity,n_iter=5000,n_jobs=-1,learning_rate='auto',init='pca',verbose=1)
X = tsne.fit_transform(df)

print(tsne.kl_divergence_)
data = pd.DataFrame(data=X,index=df.index)
data.to_csv("tsne_100_5000.csv.gz")