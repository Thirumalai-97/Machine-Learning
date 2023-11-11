from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt


data=pd.read_csv('DATA.csv')
X=data[['Feature-1','Feature-2']]

for k in range(2,6):
  print("The value of k is", k)
  kmeans = KMeans(n_clusters=k, random_state=0)
  kmeans.fit(X)
  data['cluster'] = kmeans.labels_
  plt.scatter(X['Feature-1'], X['Feature-2'], c=data['cluster'], cmap='rainbow')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title(f' number of clusters {k}')
  plt.show()


