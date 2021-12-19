# !pip3 install dit
# !pip3 install pyinform
# import dask_ml.model_selection as dcv
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
# import EEGExtract as eeg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
import sys
import csv
import os
import math
import glob
from scipy import io,signal
import numpy as np
import pandas as pd

import pickle
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import copy
from sklearn import feature_selection
# from sklearn.ensemble import RandomForestClassifier
import argparse

# import cuml
# from cuml.svm import SVR
# from cuml.ensemble import RandomForestRegressor
# from cuml.svm import SVC
# from cuml.ensemble import RandomForestClassifier
# from cuml.metrics import  accuracy_score

#Import Machine Learning Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

# import joblib
# sys.modules['sklearn.externals.joblib'] = joblib

def loadFeaturesDict(dataset):
    featuresDict = {'shannonEntropy': None,
                'ShannonRes_delta':None,
                'ShannonRes_theta':None,
                'ShannonRes_alpha':None,
                'ShannonRes_beta':None,
                'ShannonRes_gamma':None,
                'HjorthComp':None,
                'HjorthMob':None,
                'falseNearestNeighbor':None,
                'medianFreq':None,
                'bandPwr_delta':None, 
                'bandPwr_theta':None, 
                'bandPwr_alpha':None, 
                'bandPwr_beta':None, 
                'bandPwr_gamma':None,
                'stdDev':None,
                'diffuseSlowing':None,
                'spikeNum':None,
                'deltaBurstAfterSpike':None,
                'shortSpikeNum':None,
                'numBursts':None,
                'burstLenMean':None,
                'burstLenStd':None,
                'numSuppressions':None,
                'suppLenMean':None,
                'suppLenStd':None,
                'dasm_delta': None,
                'dasm_theta': None,
                'dasm_alpha': None,
                'dasm_beta': None,
                'dasm_gamma': None,
                'rasm_delta': None,
                'rasm_theta': None,
                'rasm_alpha': None,
                'rasm_beta': None,
                'rasm_gamma': None,
                }

    # featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'
    featurepath = os.getcwd() + '/Features/' + dataset + '/' #change

    featuresDict['shannonEntropy'] = np.load(featurepath + "shannonEntropy_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_delta'] = np.load(featurepath + "ShannonRes_sub_bands_delta_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_theta'] = np.load(featurepath + "ShannonRes_sub_bands_theta_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_alpha'] = np.load(featurepath + "ShannonRes_sub_bands_alpha_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_beta'] = np.load(featurepath + "ShannonRes_sub_bands_beta_1_1.npz", allow_pickle=True)['features']

    featuresDict['ShannonRes_gamma'] = np.load(featurepath + "ShannonRes_sub_bands_gamma_1_1.npz", allow_pickle=True)['features']

    #lyapunov
    # featuresDict['hFD'] = np.load(featurepath + d/DEAP_hFD.npy",allow_pickle=True)['features']

    featuresDict['HjorthComp'] = np.load(featurepath + "Hjorth_complexity_1_1.npz", allow_pickle=True)['features']

    featuresDict['HjorthMob'] = np.load(featurepath + "Hjorth_mobilty_1_1.npz",allow_pickle=True)['features']

    featuresDict['falseNearestNeighbor'] = np.load(featurepath + "falseNearestNeighbor_1_1.npz",allow_pickle=True)['features']

    featuresDict['medianFreq'] = np.load(featurepath + "medianFreq_1_1.npz",allow_pickle=True)['features']

    featuresDict['bandPwr_delta']=np.load(featurepath+"bandPwr_delta_1_1.npz", allow_pickle = True)['features']

    featuresDict['bandPwr_theta']=np.load(featurepath + "bandPwr_theta_1_1.npz", allow_pickle = True)['features']
    
    featuresDict['bandPwr_alpha']=np.load(featurepath + "bandPwr_alpha_1_1.npz", allow_pickle = True)['features']
    
    featuresDict['bandPwr_beta']=np.load(featurepath + "bandPwr_beta_1_1.npz", allow_pickle = True)['features']

    featuresDict['bandPwr_gamma']=np.load(featurepath + "bandPwr_gamma_1_1.npz", allow_pickle = True)['features']

    featuresDict['stdDev'] = np.load(featurepath + "stdDev_1_1.npz",allow_pickle=True)['features']

    # featuresDict['regularity'] = np.load(featurepath + "",allow_pickle=True)['features']

    # featuresDict['volt05'] = np.load(featurepath + "",allow_pickle=True)['features']
    # featuresDict['volt10'] = np.load(featurepath + "",allow_pickle=True)['features']
    # featuresDict['volt20'] = np.load(featurepath + "",allow_pickle=True)['features']


    featuresDict['diffuseSlowing'] = np.load(featurepath + "diffuseSlowing_1_1.npz",allow_pickle=True)['features']

    featuresDict['spikeNum'] = np.load(featurepath + "spikeNum_1_1.npz",allow_pickle=True)['features']

    featuresDict['deltaBurstAfterSpike'] = np.load(featurepath + "deltaBurstAfterSpike_1_1.npz",allow_pickle=True)['features']

    featuresDict['shortSpikeNum'] = np.load(featurepath + "shortSpikeNum_1_1.npz", allow_pickle=True)['features']

    featuresDict['numBursts'] = np.load(featurepath + "numBursts_1_1.npz",allow_pickle=True)['features']

    featuresDict['burstLenMean'] = np.load(featurepath + "burstLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

    featuresDict['burstLenStd'] = np.load(featurepath + "burstLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']

    # featuresDict['burstBandPowers'] = np.load(featurepath + "",allow_pickle=True)['features']

    featuresDict['numSuppressions'] = np.load(featurepath + "numSuppressions_1_1.npz",allow_pickle=True)['features']

    featuresDict['suppLenMean'] = np.load(featurepath + "suppressionLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

    featuresDict['suppLenStd'] = np.load(featurepath + "suppressionLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']


    featuresDict['dasm_delta'] = np.load(featurepath + "dasm_delta_1_1.npz",allow_pickle=True)['features']
    featuresDict['dasm_theta'] = np.load(featurepath + "dasm_theta_1_1.npz",allow_pickle=True)['features']
    featuresDict['dasm_alpha'] = np.load(featurepath + "dasm_alpha_1_1.npz",allow_pickle=True)['features']
    featuresDict['dasm_beta'] = np.load(featurepath + "dasm_beta_1_1.npz",allow_pickle=True)['features']
    featuresDict['dasm_gamma'] = np.load(featurepath + "dasm_gamma_1_1.npz",allow_pickle=True)['features']

    featuresDict['rasm_delta'] = np.load(featurepath + "rasm_delta_1_1.npz",allow_pickle=True)['features']
    featuresDict['rasm_theta'] = np.load(featurepath + "rasm_theta_1_1.npz",allow_pickle=True)['features']
    featuresDict['rasm_alpha'] = np.load(featurepath + "rasm_alpha_1_1.npz",allow_pickle=True)['features']
    featuresDict['rasm_beta'] = np.load(featurepath + "rasm_beta_1_1.npz",allow_pickle=True)['features']
    featuresDict['rasm_gamma'] = np.load(featurepath + "rasm_gamma_1_1.npz",allow_pickle=True)['features']

    return featuresDict

params_dict = {
    str(type(DecisionTreeRegressor()).__name__) : {
           "splitter":["best","random"],
           "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] },
    str(type(SVR()).__name__) : {
            'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
            'C' : [1,5,10],
            'degree' : [3,8],
            'coef0' : [0.01,10,0.5],
            'gamma' : ('auto','scale')},
    str(type(RandomForestRegressor()).__name__) : {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
    },
    str(type(XGBRegressor()).__name__) : {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.5, 0.7],
            'colsample_bytree': [0.5, 0.7],
            'n_estimators' : [100, 200, 500],
            'objective': ['reg:squarederror']
    },
    str(type(KNeighborsRegressor()).__name__) : {
            'n_neighbors': [2,3,4,5,6],
            'weights': ['uniform','distance']},
    
}

