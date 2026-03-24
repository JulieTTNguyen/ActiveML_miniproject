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
print('Explained variance for the components:',np.cumsum(explained_var)[:8])
plt.plot(np.array([0]+list(np.cumsum(explained_var)))[:9])
plt.title('Analysis of first 8 principal components in PCA')
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.ylim((0.5,1.05))
plt.hlines(y=0.99,xmin=0,xmax=8,colors='orange',linestyles='--',label='99% explained ')
plt.legend()
plt.savefig('figures/PCA_components.png',format='png')
