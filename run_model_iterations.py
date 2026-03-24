from sklearn.datasets import fetch_covtype
from dataloading import prepare_data
from utils import train_iteratively, uncertainty
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

### ---- Are we doing it with the even distribution? ----
even_distribution=True

### ---- Definitions ----
model=LogisticRegression(solver='lbfgs', max_iter=400)
methods=['random','margin','least_confident','entropy']
addition=1
iterations=50

### ---- Load dataset ----
cov_type = fetch_covtype()

### ---- Loop to run the analysis multiple times ----
results_total=defaultdict(list)
results_by_class=defaultdict(list)
for i in range(1):
    ## -- Get data --
    data,explained_var = prepare_data(
    cov_type,
    n_init=14,
    n_points=30000,   # optional (set None to use full dataset)
    seed=np.random.randint(0,20000000),
    even_distribution=even_distribution)


    ## -- Train for all methods--
    for method in methods:
        print(f'{method} in iteration {i}')
        test_acc,test_acc_by_class=train_iteratively(
            data=data,model=model,measure=method,addn=addition,iterations=iterations)
        
        ## -- Save results --
        results_total[method].append(test_acc)
        results_by_class[method].append(test_acc_by_class)


### ---- Save as files ----
results_total=pd.DataFrame(results_total)
results_by_class=pd.DataFrame(results_by_class)

if even_distribution:
    results_total.to_csv('results/even_total_accuracy.csv',index=False,header=False,mode='a')
    results_by_class.to_csv('results/even_class_accuracy.csv',index=False,header=False,mode='a')
else:
    results_total.to_csv('results/uneven_total_accuracy.csv',index=False,header=False,mode='a')
    results_by_class.to_csv('results/uneven_class_accuracy.csv',index=False,header=False,mode='a')
