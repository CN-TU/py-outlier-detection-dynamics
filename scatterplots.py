import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path

ind_file = sys.argv[1]
norm = sys.argv[2]

currentpath = os.path.dirname(os.path.abspath(__file__))
plotfolder = currentpath+"/plots/"+norm+"/indices"
Path(plotfolder).mkdir(parents=True, exist_ok=True)

print("Plots saved in:", plotfolder)

algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
satypes = ['sa_dim', 'sa_size', 'sa_outr', 'sa_ddif', 'sa_mdens', 'sa_clusts','sa_loc','cardio','shuttle','waveform','wilt']

metrics = ['disc_power','coherence','bias','robustness','rcvi','rcvo','adj_ap','roc_auc','P-stability','P-confidence']
metSH = ['DP','ɣ','β','φ','RCVi','RCVo','AAP','ROC','T','C']
maxy = [2.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
miny = [-1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1, -0.1]


df = pd.read_csv(ind_file)
df['idd'] = 0

for index, row in df.iterrows():
        try:
            idd = int(list(filter(str.isdigit, row['dataset']))[0])
        except:
            idd = -1
        if idd>=0:
            dataset = row['dataset'].split(str(idd))[0]
            df.loc[index,'dataset'] = dataset        
        df.loc[index,'idd'] = idd

datalg = {'x':np.zeros(10), 'i':np.arange(10)}
df_legend = pd.DataFrame(datalg)

for j,satype in enumerate(satypes):
    fig, ax = plt.subplots(1,11, figsize=(23,4), gridspec_kw={'width_ratios': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1]})
    colorbar = True
    print("Plotting...", satype)
    for k,metric in enumerate(metrics):
        axs = ax[k]
        for i,alg in enumerate(algs):
            dfm = df[df["dataset"] == satype]
            dfms = dfm[dfm["metric"] == metric]
            if len(dfms['idd'].unique())>1:
                sns.scatterplot(data=dfms, x=i, y=alg, hue="idd", ax=axs, palette='copper', legend=False)
            else:
                sns.scatterplot(data=dfms, x=i, y=alg, color='black', ax=axs, legend=False)
                colorbar = False

        axs.set_xticks(np.arange(len(algs)), algs, rotation='vertical')
        #if k==0:
            #axs.set_ylabel(satype)
        #else:
        axs.set_ylabel("")
        axs.grid(axis='x')
        #if j==0:
        axs.set_title(metSH[k])
        axs.set_ylim([miny[k],maxy[k]])

    sns.scatterplot(data=df_legend, x='x', y='i', hue='i', ax=ax[10], palette='copper', legend=False)
    ax[10].spines['top'].set_visible(False)
    ax[10].spines['right'].set_visible(False)
    ax[10].spines['bottom'].set_visible(False)
    ax[10].spines['left'].set_visible(False)
    ax[10].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    ax[10].set_xlabel("")
    ax[10].set_ylabel("experimental param. 'i'")
    #ax[10].yaxis.tick_right()
    ax[10].set_yticks(np.arange(10))
    nameplot = plotfolder + '/scatter_' + satype + '.pdf'   
    plt.tight_layout()
    plt.savefig(nameplot)
    plt.close()




