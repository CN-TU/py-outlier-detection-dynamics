"""
==============================================
Extraction of Perini's Stability and Confidence 
metrics from datasets and outlierness scores
 
FIV, Sep 2024
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
from pathlib import Path

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sdoclust import SDO
from utils.indices import get_indices
from utils.stability import *
from utils.ExCeeD import *
from hdbscan import HDBSCAN, approximate_predict_scores #GLOSH

np.random.seed(100)

class cGLOSH():
    def __init__(self):
        self.model = HDBSCAN(prediction_data=True)

    def fit(self, X):
        self.model = self.model.fit(X)
        return self

    def predict(self, X):
        y = approximate_predict_scores(self.model, X)
        return y

    def get_model_scores(self):
        return self.model.outlier_scores_

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
    model = SDO(x=6)
    return model

def glosh(c):
    model = cGLOSH()
    return model

def select_algorithm(argument,k):
    switcher = {"ABOD":abod, "HBOS":hbos, "iForest":iforest, "K-NN":knn, "LOF":lof, "OCSVM":ocsvm, "SDO":sdo, "GLOSH":glosh}
    model = switcher.get(argument, lambda: "Invalid algorithm")
    return model(k)

def normalize(tr, ts, method, a):
    if (method=='gauss' and a=='ABOD'):
        tr = -1 * np.log10(tr/np.max(tr))
        ts = -1 * np.log10(ts/np.max(tr))
    if method=='gauss':
        mu = np.nanmean(tr)
        sigma = np.nanstd(tr)
        tr = (tr - mu) / (sigma * np.sqrt(2))
        tr = erf(tr)
        tr = tr.clip(0, 1).ravel()
        ts = (ts - mu) / (sigma * np.sqrt(2))
        ts = erf(ts)
        ts = ts.clip(0, 1).ravel()
    elif method=='minmax':
        trmin, trmax = tr.min(), tr.max()
        tr = (tr - trmin) / (trmax - trmin)
        ts = (ts - trmin) / (trmax - trmin)
    return tr, ts

inpath  = sys.argv[1]
norm = sys.argv[2]
skip_header = 1

currentpath = os.path.dirname(os.path.abspath(__file__))
pffolder = currentpath+"/performances"
Path(pffolder).mkdir(parents=True, exist_ok=True)
stabfile = pffolder+"/peri_stab_"+norm+".csv"
conffile = pffolder+"/peri_conf_"+norm+".csv"

algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]
cols = ["dataset","ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","GLOSH"]

print("\nData folder:",inpath)

df_stab = pd.DataFrame(columns=cols)
df_conf = pd.DataFrame(columns=cols)

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*'))):
    print("\nData file", filename)
    print("Data file index: ", idf)

    d_name = ntpath.basename(filename)
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=skip_header)
    X, ygt = dataset[:,0:-1].astype(float), dataset[:,-1].astype(int)
    
    if -1 in np.unique(ygt):
        ygt[ygt>-1] = 0
        ygt[ygt==-1] = 1
    if len(np.unique(ygt))>2:
        ygt[ygt>0] = 1

    X = MinMaxScaler().fit_transform(X)

    n_samples = len(ygt)
    outliers_fraction = sum(ygt)/len(ygt)

    stab, conf = {},{}
    stab['dataset'] = d_name
    conf['dataset'] = d_name

    ### OUTLIER DET. ALGORITHMS 
    for a_name in algs:

        print("-----------------------------")
        print("Algorithm:", a_name)

        X_train, X_test, y_train, y_test = train_test_split(X, ygt, test_size=0.1, random_state=42, stratify=ygt)

        algorithm = select_algorithm(a_name,outliers_fraction)

        if a_name == "GLOSH":
            algorithm = algorithm.fit(X_train)
            train_scores = algorithm.get_model_scores()
            test_scores = algorithm.predict(X_test)
            out_frac = np.sum(y_test)/len(y_test)
            threshold = np.quantile(test_scores, 1-out_frac)
            prediction = (test_scores > threshold)*1
        else:
            algorithm.fit(X_train)
            if a_name == "SDO":
                train_scores = algorithm.predict(X_train)
                test_scores = algorithm.predict(X_test)
                out_frac = np.sum(y_test)/len(y_test)
                threshold = np.quantile(test_scores, 1-out_frac)
                prediction = (test_scores > threshold)*1
            else:
                y_train = algorithm.predict(X_train)
                train_scores = algorithm.decision_function(X_train)
                prediction = algorithm.predict(X_test)
                test_scores = algorithm.decision_function(X_test)

        train_scores, test_scores = normalize(train_scores, test_scores, norm, a_name)

        stab_unif, inst_unif = stability_measure(X_train, X_test, algorithm, outliers_fraction, test_scores, unif = True, iterations=100, subset_low=0.3, subset_high=0.6)
        print("Stability:", stab_unif, inst_unif)
        confidence = ExCeeD(train_scores, test_scores, prediction, outliers_fraction)
        num_outliers, m = sum(ygt), len(ygt)
        #confidence = np.nanmean(np.sort(confidence)[:num_outliers])
        #confidence = (confidence - num_outliers/m) / (1 - num_outliers/m)
        confidence = np.quantile(confidence,0.01)
        print("Confidence:", confidence)

        stab[a_name] = stab_unif
        conf[a_name] = confidence

    df_stab = pd.concat([df_stab, pd.DataFrame([stab])], ignore_index=True)
    df_conf = pd.concat([df_conf, pd.DataFrame([conf])], ignore_index=True)

    print(df_stab)
    print(df_conf)

if os.path.exists(stabfile):
    df_stab.to_csv(stabfile, mode='a', header=False)
else:
    df_stab.to_csv(stabfile)
print('Stability scores saved in:',stabfile)

if os.path.exists(conffile):
    df_conf.to_csv(conffile, mode='a', header=False)
else:
    df_conf.to_csv(conffile)
print('Confidence scores saved in:',conffile)



