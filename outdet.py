"""
==============================================
Comparison of anomaly detection algorithms 
with multi-dimensional data with GT
 
FIV, Dec 2022
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

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from scipy.special import erf

from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.ocsvm import OCSVM
from pyod.models.cof import COF
from pyod.models.sod import SOD
from pysdo import SDO
from PyNomaly import loop
from hdbscan import HDBSCAN #GLOSH

from indices import get_indices

np.random.seed(100)

def abod(c):
    model = ABOD(contamination=c, n_neighbors=20, method='fast')
    return model
 
def hbos(c):
    model = HBOS(contamination=c,n_bins=20)
    return model

def iforest(c):
    model = IForest(contamination=c, random_state=100)
    return model

def knn(c):
    model = KNN(contamination=c, n_neighbors=20)
    return model

def lof(c):
    model = LOF(contamination=c, n_neighbors=20)
    return model

def ocsvm(c):
    model = OCSVM(contamination=c)
    return model

def sdo(c):
    model = SDO(contamination=c, return_scores=True)
    return model

def LoOP(c):
    model = loop
    return model

def glosh(c):
    model = HDBSCAN()
    return model

def select_algorithm(argument,k):
    switcher = {"ABOD":abod, "HBOS":hbos, "iForest":iforest, "K-NN":knn, "LOF":lof, "OCSVM":ocsvm, "SDO":sdo, "LOOP":LoOP, "GLOSH":glosh}
    model = switcher.get(argument, lambda: "Invalid algorithm")
    return model(k)

def normalize(s, method):
    if method=='abodreg':
        s = -1 * np.log10(s/np.max(s))
    if (method=='gauss' or method=='abodreg'):
        mu = np.nanmean(s)
        sigma = np.nanstd(s)
        s = (s - mu) / (sigma * np.sqrt(2))
        s = erf(s)
        s = s.clip(0, 1).ravel()
    elif method=='minmax':
        s = (s - s.min()) / (s.max() - s.min())
    return s

inpath  = sys.argv[1]
scfolder = sys.argv[2]
perffile = sys.argv[3]
norm = sys.argv[4]
skip_header = int(sys.argv[5])

#algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","LOOP","GLOSH"]
#cols = ["dataset","ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","LOOP","GLOSH"]
algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
cols = ["dataset","ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
metrics = ["adj_Patn", "adj_maxf1", "adj_ap", "roc_auc", "AMI"]

currentpath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(currentpath, scfolder) 
if os.path.exists(path):
    pass
else: 
    os.mkdir(path)    

print("\nData folder:",inpath)
print("Scores folder:",scfolder)
print("Performance file:",perffile)

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*'))):
    print("\nData file", filename)
    print("Data file index: ", idf)

    dfsc = pd.DataFrame(columns=algs)
    dfpf = pd.DataFrame(columns=cols,index=metrics)

    d_name = ntpath.basename(filename)
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=skip_header)
    X, ygt = dataset[:,0:-1], dataset[:,-1].astype(int)
    
    if -1 in np.unique(ygt):
        ygt[ygt>-1] = 0
        ygt[ygt==-1] = 1
    if len(np.unique(ygt))>2:
        ygt[ygt>0] = 1

    X = MinMaxScaler().fit_transform(X)

    n_samples = len(ygt)
    outliers_fraction = sum(ygt)/len(ygt)

    ### OUTLIER DET. ALGORITHMS 
    for a_name in algs:

        print("-----------------------------")
        print("Algorithm:", a_name)

        algorithm = select_algorithm(a_name,outliers_fraction)
        if a_name == "LOOP":
            scores = algorithm.LocalOutlierProbability(X, extent=2, n_neighbors=20, use_numba=True).fit().local_outlier_probabilities.astype(float)
            scores = normalize(scores, norm)
            threshold = np.quantile(scores, 1-outliers_fraction)
            y = (scores > threshold)*1
        elif a_name == "GLOSH":
            algorithm.fit_predict(X)
            scores = algorithm.outlier_scores_
            scores = normalize(scores, norm)
            threshold = np.quantile(scores, 1-outliers_fraction)
            y = (scores > threshold)*1
        else:        
            algorithm.fit(X)
            if a_name == "SDO":
                scores = algorithm.predict(X)
                scores = normalize(scores, norm)
                threshold = np.quantile(scores, 1-outliers_fraction)
                y = (scores > threshold)*1
            else:
                y = algorithm.predict(X)
                scores = algorithm.decision_scores_
                if (a_name == "ABOD" and norm == "gauss"):
                    scores = normalize(scores, "abodreg")
                else:
                    scores = normalize(scores, norm)

        AMI = adjusted_mutual_info_score(ygt, y)
        RES = get_indices(ygt, scores)

        dfpf.loc['adj_Patn', a_name] = RES['adj_Patn']
        dfpf.loc['adj_maxf1', a_name] = RES['adj_maxf1']
        dfpf.loc['adj_ap', a_name] = RES['adj_ap']
        dfpf.loc['roc_auc', a_name] = RES['auc']
        dfpf.loc['AMI', a_name] = AMI
        
        print("Adj P@n: ", RES['adj_Patn'])
        print("Adj MaxF1: ", RES['adj_maxf1'])
        print("Adj AP: ", RES['adj_ap'])
        print("Adj ROC-AUC: ", RES['auc'])
        print("Adj AMI: ", AMI)
        print("-----------------------------\n")

        dfsc[a_name] = scores

    dfpf['dataset']=d_name
    dfsc.to_csv(scfolder+'/'+d_name, index=False)
    print('Scores saved in:',(scfolder+'/'+d_name))

    if os.path.exists(perffile):
        dfpf.to_csv(perffile, mode='a', header=False)
    else:
        dfpf.to_csv(perffile)

    print('Peformances saved in:',perffile)

