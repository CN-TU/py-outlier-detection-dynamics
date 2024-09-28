"""
==============================================
Comparison of outlierness scores 
with multi-dimensional data with GT
 
FIV, Jan 2024
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import glob
import os
import re
import ntpath
import matplotlib.pyplot as plt
import matplotlib
import csv
from pathlib import Path

from scipy import stats

def coherence(dX, dy):
    ms = np.zeros(len(np.unique(dy)))
    ss = np.zeros(len(np.unique(dy)))
    o = 0
    for y in np.unique(dy):
        ms[y] = np.mean(dX[dy==y])
        ss[y] = 2*np.std(dX[dy==y])
    rangey = len(np.unique(dy))-1
    for i in range(rangey):
        b = np.max([ms[i]+ss[i],ms[i+1]+ss[i+1]]) - np.min([ms[i]-ss[i],ms[i+1]-ss[i+1]])
        a = np.min([ms[i]+ss[i],ms[i+1]+ss[i+1]]) - np.max([ms[i]-ss[i],ms[i+1]-ss[i+1]])
        if a<0:
            c = 0
        else:
            c = a/b
        o += c 
    return 1-o/rangey

def OI_variance(dX, dy, mode='mean'):
    ntop = np.sum(dy)
    inls = dX[:len(dX)-ntop]
    outs = dX[len(dX)-ntop:]
    if mode=='median':
        q3, q1 = np.percentile(inls, [75 ,25])
        iqr_inls = (q3 - q1)/np.nanmedian(inls)
        q3, q1 = np.percentile(outs, [75 ,25])
        iqr_outs = (q3 - q1)/np.nanmedian(outs)
        return iqr_outs, iqr_inls
    elif mode=='madm':
        madm_inls = stats.median_abs_deviation(inls, scale=1)/np.nanmedian(inls)
        madm_outs = stats.median_abs_deviation(outs, scale=1)/np.nanmedian(outs)
        return madm_outs, madm_inls
    else:
        var_inls = np.nanstd(inls)/np.nanmean(inls)
        var_outs = np.nanstd(outs)/np.nanmean(outs)
        return var_outs, var_inls

def discriminant_power(dX):
    return np.log10(1+stats.kurtosis(dX, axis=0,bias=False))


data_folder = sys.argv[1]
scores_folder  = sys.argv[2]
norm  = sys.argv[3]
skip_header = 1

currentpath = os.path.dirname(os.path.abspath(__file__))
plots_folder = currentpath+"/plots/"+norm+"/S_curves"
pffolder = currentpath+"/"+"/performances"
Path(plots_folder).mkdir(parents=True, exist_ok=True)
Path(pffolder).mkdir(parents=True, exist_ok=True)
dyn_file = pffolder+"/dynamic_"+norm+".csv"

algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
cols = ["dataset","ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
if "real_data" in data_folder:
    satypes = ['cardio','shuttle','waveform','wilt']
else:
    satypes = ['sa_dim', 'sa_size', 'sa_outr', 'sa_ddif', 'sa_mdens', 'sa_clusts','sa_loc']

performance = {}

print("\nData folder:",data_folder)
print("Scores folder:",scores_folder)

sdo,abod,hbos,iforest,knn,lof,ocsvm,loop,glosh =  np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10)),np.zeros((1000,10))
models = [sdo,abod,hbos,iforest,knn,lof,ocsvm,glosh]

cmap = matplotlib.cm.get_cmap('copper')
normplot = matplotlib.colors.Normalize(vmin=-1, vmax=10)

for satype in satypes:

    fig, ax = plt.subplots(1,len(algs), figsize=(23,4))

    for idf, filename in enumerate(glob.glob(os.path.join(data_folder, satype+"*"))):
        print("\nData file: ", filename)

        d_name = ntpath.basename(filename)
        idd = int(list(filter(str.isdigit, d_name))[0])
        print("Data file index: ", idd)
        dataset = np.genfromtxt(filename, delimiter=',', skip_header=skip_header)
        X, ygt = dataset[:,0:-1], dataset[:,-1].astype(int)
        
        if -1 in np.unique(ygt):
            ygt[ygt>-1] = 0
            ygt[ygt==-1] = 1

        scfile = scores_folder + '/' + d_name 
        print("Scores file: ", scfile)

        dfsc = pd.read_csv(scfile)

        m = ygt.size
        num_outliers = np.sum(ygt)
        dfsc_norm = (dfsc-dfsc.min())/(dfsc.max()-dfsc.min())
        dfsc_norm['y'] = ygt

        dfsc_norm.sort_values(by=['y'], inplace=True)

        x = np.linspace(0,1,len(dfsc_norm))
        hop = len(dfsc_norm)/1000
        xc = np.arange(0, 1000*hop, hop).astype(int)[:1000]

        color = cmap(normplot(idd))

        for i,alg in enumerate(algs):         
            dfsc_norm.sort_values(by=[alg], inplace=True)
            dfsc_norm['x']=x
            if i==0:
                dfsc_norm.plot(x='x', y=alg, ax=ax[i], title=alg, c=color, xlabel='', ylabel='normalized outlierness', legend=False)
            else:
                dfsc_norm.plot(x='x', y=alg, ax=ax[i], title=alg, c=color, xlabel='', ylabel='', legend=False)
            
            dX = dfsc_norm[alg].to_numpy()
            dy = dfsc_norm['y'].to_numpy()
            performance[(d_name, alg, 'disc_power')] = discriminant_power(dX)
            performance[(d_name, alg, 'rcvo')], performance[(d_name, alg, 'rcvi')] = OI_variance(dX,dy, 'madm')
            performance[(d_name, alg, 'bias')] = np.median(dX)
            performance[(d_name, alg, 'coherence')] = coherence(dX, dy)

            models[i][:,idd]=dX[xc]
                    
    ax2 = fig.add_axes([0.91, 0.12, 0.015, 0.75])
    bounds = np.arange(0,11)
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=normplot, spacing='proportional', ticks=bounds+0.5, boundaries=bounds, format='%1i')

    for i,m in enumerate(models):
        m=m+1
        performance[(satype, algs[i], 'robustness')] = 1 - np.sqrt(np.nanmean( np.nanstd(m,axis=1)*np.nanstd(m,axis=1)/(np.nanmax(m,axis=1)-np.nanmin(m,axis=1)) ))
    plotname = plots_folder + '/group_'+ satype + '.pdf' 
    plt.savefig(plotname, bbox_inches='tight')
    

df = pd.DataFrame(columns=["metric","dataset","algorithm","val"])

for key, val in performance.items():
    d,a,m = key
    df.loc[len(df)] = [m,d,a,val]   

df = df.pivot_table(index=['metric', 'dataset'], columns='algorithm', values='val', aggfunc='first').reset_index()

if os.path.exists(dyn_file):
    df.to_csv(dyn_file, mode='a', header=False)
else:
    df.to_csv(dyn_file)



print("S-curve plots saved in:", plots_folder)
print("Dynamic performances saved in:", dyn_file)
