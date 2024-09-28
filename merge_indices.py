import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

dyna_file = sys.argv[1]
perf_file = sys.argv[2]
stab_file = sys.argv[3]
confi_file = sys.argv[4]
norm = sys.argv[5]

currentpath = os.path.dirname(os.path.abspath(__file__))
pffolder = currentpath+"/performances"
Path(pffolder).mkdir(parents=True, exist_ok=True)
out_file = pffolder+"/all_"+norm+".csv"

algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
satypes = ['sa_dim', 'sa_size', 'sa_outr', 'sa_ddif', 'sa_mdens', 'sa_clusts','sa_loc']

dfd = pd.read_csv(dyna_file, index_col=0)
dfp = pd.read_csv(perf_file)
dfp.rename(columns={dfp.columns[0]: "metric"}, inplace = True)

dfs = pd.read_csv(stab_file, index_col=0)
dfc = pd.read_csv(confi_file, index_col=0)

dfs['metric'] = "P-stability"
dfc['metric'] = "P-confidence"

df = pd.concat([dfd, dfp, dfs, dfc])
df = df.reset_index(drop=True)

df.to_csv(out_file, index=False) 
print("Indices merged and saved in:", out_file)
#df.to_csv(out_file, index=False, float_format="%.3f")  


