import numpy as np
import pandas as pd
import sys
from pathlib import Path

minmax_file = sys.argv[1]
proba_file = sys.argv[2]
latexfile = sys.argv[3]

algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
satypes = ['sa_dim', 'sa_size', 'sa_outr', 'sa_ddif', 'sa_mdens', 'sa_clusts','sa_loc']
selected_metrics = ['disc_power','rcvi','rcvo','coherence','bias','robustness','adj_ap','roc_auc','P-stability','P-confidence']
normalizations = ['linear','Gaussian']

dfm = pd.read_csv(minmax_file)
dfp = pd.read_csv(proba_file)

dfm['norm'] = 'linear'
dfp['norm'] = 'Gaussian'

df = pd.concat([dfm,dfp], ignore_index=True)

for index, row in df.iterrows():
        try:
            idd = int(list(filter(str.isdigit, row['dataset']))[0])
        except:
            idd = -1
        if idd>=0:
            dataset = row['dataset'].split(str(idd))[0]
            df.loc[index,'dataset'] = dataset        
        df.loc[index,'idd'] = idd

df_out = pd.DataFrame(columns=['dataset','algorithm','norm.','min/max']+selected_metrics)

for satype in satypes:
    aux = df[df["dataset"] == satype]
    for n in normalizations:
        auxb = aux[aux["norm"] == n]
        for alg in algs:
            max_row = [satype, alg, n, 'max']
            min_row = [satype, alg, n, 'min']
            for metric in selected_metrics:              
                auxc = auxb[auxb["metric"] == metric]
                vals = auxc[alg].to_numpy()
                if len(vals) == 0:
                    max_row.append(np.nan)
                    min_row.append(np.nan)
                else:
                    maxv = np.nanmax(vals)
                    minv = np.nanmin(vals)
                    max_row.append(maxv)
                    min_row.append(minv)
                print(satype, n, alg, metric)
            df_out.loc[len(df_out)] =  min_row
            df_out.loc[len(df_out)] =  max_row

df_out.to_latex(latexfile, index=False, float_format="%.2f")  


