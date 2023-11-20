import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt


data_file = sys.argv[1]
scores_file  = sys.argv[2]
skip_header = int(sys.argv[3])

algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]

print("\nData file:",data_file)
print("Scores file:",scores_file)

data_to_print = 2000

dataset = np.genfromtxt(data_file, delimiter=',', skip_header=skip_header)
X, ygt = dataset[:,0:-1], dataset[:,-1].astype(int)
        
if -1 in np.unique(ygt):
    ygt[ygt>-1] = 0
    ygt[ygt==-1] = 1

dfsc = pd.read_csv(scores_file)
m = ygt.size
num_outliers = np.sum(ygt)

fig, ax = plt.subplots(2,4, figsize=(20,10))

for i,alg in enumerate(algs):
    p = dfsc[alg].to_numpy()
    a,b = np.unravel_index(i, (2, 4))
    ax[a,b].scatter(X[0:data_to_print,0],X[0:data_to_print,1], s=1, cmap='coolwarm', c=p[0:data_to_print])

plt.show()
