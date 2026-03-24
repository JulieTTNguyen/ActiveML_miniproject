from sklearn.datasets import fetch_covtype
from dataloading import prepare_data
import numpy as np
import matplotlib.pyplot as plt

## ---- Load data ----

cov_type = fetch_covtype()

data, explained_var = prepare_data(
    cov_type,
    n_init=20,
    n_points=20000,   # optional (set None to use full dataset)
    seed=42,
    n_components=None # Set to None to use all components
)

## ---- Plot components ----
print([0]+list(np.cumsum(explained_var)))
plt.plot(np.array([0]+list(np.cumsum(explained_var))))
plt.title('Analysis of Components in PCA')
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.ylim((0.0,1.1))
plt.show()