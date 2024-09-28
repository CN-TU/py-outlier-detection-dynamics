import numpy as np
import pandas as pd
import sys

minmax_file = sys.argv[1]
proba_file = sys.argv[2]
plot_folder = sys.argv[3]

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

df_out = pd.DataFrame(columns=['dataset','algorithm','norm.','i']+selected_metrics)

for satype in satypes:
    aux = df[df["dataset"] == satype]
    for n in normalizations:
        auxb = aux[aux["norm"] == n]
        for alg in algs:
            for i in range(10):
                row = [satype, alg, n, i]
                for metric in selected_metrics:              
                    auxc = auxb[auxb["metric"] == metric]
                    vals = auxc[alg].to_numpy()
                    #print(metric, len(vals))
                    try: 
                        row.append(vals[i])
                    except:
                        row.append(np.nan)

                df_out.loc[len(df_out)] =  row

dflin = df_out.loc[df_out['norm.'] == "linear"].select_dtypes(include=np.number)
dfgauss = df_out.loc[df_out['norm.'] == "Gaussian"].select_dtypes(include=np.number)

dflin.drop(['i'], inplace=True, axis=1)
dflin.rename(columns = {'disc_power':'DP', 'rcvi':'RCVi', 'rcvo':'RCVo', 'coherence':'ɣ', 
        'bias':'β', 'robustness':'φ', 'adj_ap':'AAP', 'roc_auc':'ROC', 'P-stability':'T',
        'P-confidence':'C'}, inplace = True)


dfgauss.drop(['i'], inplace=True, axis=1)
dfgauss.rename(columns = {'disc_power':'DP', 'rcvi':'RCVi', 'rcvo':'RCVo', 'coherence':'ɣ', 
        'bias':'β', 'robustness':'φ', 'adj_ap':'AAP', 'roc_auc':'ROC', 'P-stability':'T',
        'P-confidence':'C'}, inplace = True)


import matplotlib.pyplot as plt
import seaborn as sns

dfcorr = dflin.corr()
f, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(dfcorr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(dfcorr, mask=mask, annot=True, cmap=cmap, fmt=".2f", center=0, square=True, linewidths=.5)#, cbar_kws={"shrink": .5})
plt.yticks(rotation=0) 
plt.tight_layout()
plt.savefig(plot_folder+"/corr_lin.pdf")
plt.close()


dfcorr = dfgauss.corr()
f, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(dfcorr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(dfcorr, mask=mask, annot=True, cmap=cmap, fmt=".2f", center=0, square=True, linewidths=.5)#, cbar_kws={"shrink": .5})
plt.yticks(rotation=0) 
plt.tight_layout()
plt.savefig(plot_folder+"/corr_gauss.pdf")
plt.close()
