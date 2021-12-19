from ImportUtils import *
from sklearn.model_selection import ParameterGrid
# import EEGExtract as eeg

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# from sklearn.svm import SVC
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as sklearnrfi

import os
import glob
from scipy import io,signal
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


import matplotlib.pyplot as plt
import seaborn as sns
import copy

# def on_step(optim_result):
#     """
#     Callback meant to view scores after
#     each iteration while performing Bayesian
#     Optimization in Skopt"""
#     score = forest_bayes_search.best_score_
#     print("best score: %s" % score)
#     if score < 1.6:
#         print('Interrupting!')
#         return True




def topElectrodeRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, ht = False):
    '''
    Ranks of features according to rmse computed by regressor passed in clf
    Plots electrode v/s rmse graph
    
    '''
    pwd = os.getcwd()

    #Load extracted features
    #####################################################################################################################################################
    # featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'
    featurepath = os.getcwd() + '/Features/' + dataset + '/'
    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']

    output_file = os.getcwd() + '/Output/' + dataset + '/' + "EC_{}_{}_{}.csv".format(type(clf).__name__,label,  "ht" if ht else "")
    with open(output_file,'a') as fd:
        fd.write("Dataset used: {}\n".format(dataset))
        fd.write("ML Model used: {}\n".format(clf))
        

    rmseList = []
    # electrodeList = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1','AF4','F4','F8','FC6','T8','P8','O2']
    electrodeList = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    fs = sfreq
    pwd = os.getcwd()

    
    featuresDict = loadFeaturesDict(dataset)
    asm_features = ['dasm_delta', 'dasm_theta', 'dasm_alpha', 'dasm_beta', 'dasm_gamma', 'rasm_delta', 'rasm_theta', 'rasm_alpha', 'rasm_beta', 'rasm_gamma']
    for asm in asm_features:
        featuresDict.pop(asm)

    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)

    for k in list(featuresDict.keys()):
        if k not in common:
            # pop out common feature
            featuresDict.pop(k)


    
    selectFeatures = list(featuresDict.keys())
    #change 
    with open(output_file,'a') as fd:
        fd.write("Features List:\n")
        writer = csv.writer(fd)
        writer.writerow(selectFeatures)

    y = Y_epoch[:,label] #valence


    #####################################################################################################################################################
    
    for electrode in range(14):
        # Load FeaturesDict from memory
        

        print("Number of segments are: {}".format(ans.shape[1]))
        
        featureMatrix = np.empty((len(selectFeatures),ans.shape[1])) #[14*32 + 1,80640]
        i=0
        for key,value in featuresDict.items():
            featureMatrix[i,:] = value[electrode,:]
            i = i+1

        print(featureMatrix.T.shape)
        featureMatrix = featureMatrix.astype(np.float32)

        #Impute NaN values with zero
        if np.isnan(featureMatrix).any():
            featureMatrix = np.nan_to_num(featureMatrix,nan=0)

        #Name Feature vector columns
        feature_channel_index = []
        for feature in selectFeatures:
            feature_channel_index.append(feature + str(electrode))

        print("Number of Feature-Columns: {}\n".format(len(feature_channel_index))) #debug
        
        #Preparing dataset from feature matrix
        X = pd.DataFrame(featureMatrix.T)
        X.columns = feature_channel_index
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        

        print("Features Ready for undergoing selection tests done ...\n")

        # Perform train_test_split to get training and test data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        rmseList.append(rmse)
        

    #rank electrodes based on RMSE computed by the classifier
    electrode_df = pd.DataFrame(electrodeList)
    rmse_df = pd.DataFrame(rmseList)
    #concat two dataframes for better visualization 
    electrodeRanking = pd.concat([electrode_df, rmse_df],axis=1)
    electrodeRanking.columns = ['Electrode','RMSE']  #naming the dataframe columns
    features_result = electrodeRanking.sort_values('RMSE')
    #test # features_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "CommonElectrodeRegressionRanking" + str(window) + str(stride) + ".csv")
    print(features_result)
    # change
    with open(output_file,'a') as fd:
        fd.write("Electrode Ranking \n\n")
        
    features_result.to_csv(output_file, mode='a', header=True)

    
    ##################################################################################
    N =  features_result.shape[0]
    topRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]

    
    for n in range(1,N+1):
        

        topnelectrodes = features_result.head(n)
        electrode_index = topnelectrodes.index
        electrode_index = list(electrode_index)[:n]

        # X-Values
        featureMatrix = np.empty((len(selectFeatures)*len(electrode_index),ans.shape[1]))

        i = 0
        for index in electrode_index:
            for key,value in featuresDict.items():
                featureMatrix[i,:] = value[index,:]
                i = i+1
                
            # i = i+1
        
        featureMatrix = featureMatrix.astype(np.float32)
        print(featureMatrix.T.shape)
        
        # Removing NaN Values
        if np.isnan(featureMatrix).any():
            featureMatrix = np.nan_to_num(featureMatrix,nan=0)

        # Name Feature vector columns
        feature_channel_index = []
        for index in electrode_index:
            for feature in selectFeatures:
                feature_channel_index.append(feature + str(index))

        print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

        X = pd.DataFrame(featureMatrix.T)
        X.columns = feature_channel_index
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        

        print("Features Ready for undergoing selection tests done ...\n")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier
        y_predict = None
        if ht == True:
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
            grid_search=GridSearchCV(estimator=clf,param_grid=params_dict[str(type(clf).__name__)],cv=cv, verbose=5)
            grid_search.fit(X_train,y_train)
            best_clf = grid_search.best_estimator_
            print(best_clf)
            y_predict = grid_search.predict(X_test)
        else:
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)

        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        topRmseList.append(rmse)



    # features_result = features_result.reset_index()
    topNElectrode_df = pd.DataFrame(topNList)
    topNRmse_df = pd.DataFrame(topRmseList)
    #concat two dataframes for better visualization 
    topNElectrodeRanking = pd.concat([topNElectrode_df, topNRmse_df],axis=1)
    topNElectrodeRanking.columns = ['Electrode','RMSE']  #naming the dataframe columns
    print(topNElectrodeRanking)
    #test # topNElectrodeRanking.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topCommonElectrodeRegressionRanking" + str(window) + str(stride) + ".csv")
    # return features_result

    with open(output_file,'a') as fd:
        fd.write("Top N Electrodes Ranking\n")

    topNElectrodeRanking.to_csv(output_file, mode='a', header=True)
    
    
    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('Top N Electrodes')
    plt.ylabel('RMSE')
    plt.plot(topNElectrodeRanking.loc[:,"Electrode"], topNElectrodeRanking.loc[:,"RMSE"])
    plt.tight_layout()
    # plt.savefig(pwd + "/" + dataset + "/plots/" + "topElectrodeRegressionRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    # plt.show()
    # plt.clf()

    
def topFeaturesRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, ht = False):
    '''
    Ranks of features according to rmse computed by regressor passed in clf
    Plots electrode v/s rmse graph
    
    '''
    fs = sfreq
    pwd = os.getcwd()
    #change
    featurepath = os.getcwd() + '/Features/' + dataset + '/'    

    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']
    print("Number of segments are: {}".format(ans.shape[1]))
    #change
    output_file = os.getcwd() + '/Output/' + dataset + '/' + "FC_{}_{}_{}.csv".format(type(clf).__name__,label, "ht" if ht else "")
    with open(output_file,'a') as fd:
        fd.write("Dataset used: {}\n".format(dataset))
        fd.write("ML Model used: {}\n".format(clf))
        fd.write("Number of segments are: {}\n".format(ans.shape[1]))

    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)

    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)

    for k in list(featuresDict.keys()):
        if k not in common:
            # pop out common feature
            featuresDict.pop(k)

    featuresList = list(featuresDict.keys())
    
    y = Y_epoch[:,label] #valence

    
    rmseList = []

    ####################################################################
    #modify featuresList
    featureMatrix = np.empty((0,ans.shape[1])) #[14*32 + 1,80640]
    for key,value in featuresDict.items():
        featureMatrix = np.append(featureMatrix,value,axis=0)


    if np.isnan(featureMatrix).any():
        featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    featureMatrix = featureMatrix.astype('float64')


    feature_channel_index = []
    for feature in featuresList:
        for i in range(featuresDict[feature].shape[0]):
            if(i>=10):
                feature_channel_index.append(feature + str(i))
            else:
                feature_channel_index.append(feature + '0' + str(i))
    #change
    print(len(list(featuresDict.keys())))
    print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

    X = pd.DataFrame(featureMatrix.T)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X.columns = feature_channel_index

    #Remove Variance = 0 features     
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X)
    constant_columns = [column for column in X.columns
                    if column not in
    X.columns[constant_filter.get_support()]]
    X = constant_filter.transform(X)
    
    for column in constant_columns:
        feature_channel_index.remove(column)

    print(len(feature_channel_index),feature_channel_index )

    X = pd.DataFrame(X)
    X.columns = feature_channel_index


    filtered_featuresList = []
    print(type(X))
    for col in X.columns:
        feature = col[:-2]
        electrode = int(col[-2:])
        if(feature not in filtered_featuresList):
            filtered_featuresList.append(feature)
        
    featuresList = filtered_featuresList

    with open(output_file,'a') as fd:
        fd.write("Features List:\n")
        writer = csv.writer(fd)
        writer.writerow(featuresList)


    for feature in featuresList:
        # Load FeaturesDict from memory
        featureMatrix = featuresDict[feature]
        featureMatrix = featureMatrix.astype(np.float32)

        if np.isnan(featureMatrix).any():
            featureMatrix = np.nan_to_num(featureMatrix,nan=0)

        
        feature_channel_index = []
        
        for i in range(featuresDict[feature].shape[0]):
            feature_channel_index.append(feature + str(i))

        print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

        X = pd.DataFrame(featureMatrix.T)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        X.columns = feature_channel_index
        

        print("Features Ready for undergoing selection tests done ...\n")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier
        # clf = xgb.XGBClassifier(verbose = 5)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        rmseList.append(rmse)
        

    
    features_df = pd.DataFrame(featuresList)
    rmse_df = pd.DataFrame(rmseList)
    #concat two dataframes for better visualization 
    featureRanking = pd.concat([features_df, rmse_df],axis=1)
    featureRanking.columns = ['Feature','RMSE']  #naming the dataframe columns
    features_result = featureRanking.sort_values('RMSE')
    # features_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "CommonFeaturesRegressionRanking" + str(window) + str(stride) + ".csv")
    print(features_result)
    with open(output_file,'a') as fd:
        fd.write("Feature Ranking \n")
    features_result.to_csv(output_file, mode='a', header=True)
    
    # features_result = features_result.reset_index()
        ###########################################
    N =  features_result.shape[0]
    topNRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]


    
    for n in range(1,N+1):
        

        topnfeatures = copy.deepcopy(features_result.head(n))
        topnfeatures = topnfeatures['Feature'].tolist() #list of feature-names
        
        # X-Values################################################

        featureMatrix = np.empty((0,ans.shape[1]))
    
        for feature in topnfeatures:
            featureMatrix = np.append(featureMatrix, featuresDict[feature], axis=0)
        
        featureMatrix = featureMatrix.astype(np.float32)
        print(featureMatrix.T.shape)

        feature_channel_index = []
        for feature in topnfeatures:
            i=0
            for i in range(featuresDict[feature].shape[0]):
                feature_channel_index.append(feature + str(i))

        
        # Removing NaN Values
        if np.isnan(featureMatrix).any():
            featureMatrix = np.nan_to_num(featureMatrix,nan=0)

        
        print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

        X = pd.DataFrame(featureMatrix.T)
        X.columns = feature_channel_index
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        

        print("Features Ready for undergoing selection tests done ...\n")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)


        y_predict = None
        if ht == True:
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
            grid_search=GridSearchCV(estimator=clf,param_grid=params_dict[str(type(clf).__name__)],cv=cv, verbose=5)
            grid_search.fit(X_train,y_train)
            best_clf = grid_search.best_estimator_
            print(best_clf)
            y_predict = grid_search.predict(X_test)
        else:
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)

        rmse = mean_squared_error(y_test, y_predict,squared=False)
        print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        topNRmseList.append(rmse)




    # features_result = features_result.reset_index()
    topNFeatures_df = pd.DataFrame(topNList)
    topNRmse_df = pd.DataFrame(topNRmseList)

    #concat two dataframes for better visualization 
    topNFeaturesRanking = pd.concat([topNFeatures_df, topNRmse_df],axis=1)
    topNFeaturesRanking.columns = ['Feature','RMSE']  #naming the dataframe columns
    print(topNFeaturesRanking)
    # topNFeaturesRanking.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topCommonFeaturesRegressionRanking" + str(window) + str(stride) + ".csv")
    with open(output_file,'a') as fd:
        fd.write("Top N Features Ranking\n")
    topNFeaturesRanking.to_csv(output_file, mode='a', header=True)    
    
    
    
    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(25, 10)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('Top N Features')
    plt.ylabel('RMSE')
    # plt.title("Top N Features v/s RMSE Plot for Window:{} Stride:{} epoched data by varying N".format(window,stride))
    plt.plot(topNFeaturesRanking.loc[:,"Feature"], topNFeaturesRanking.loc[:,"RMSE"])
    plt.tight_layout()
    # plt.savefig(pwd + "/" + dataset + "/plots/" + "topFeaturesRegressionRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight', dpi = 300)
    # plt.show()
    # plt.clf()


    
    
def topFeatureColumnsRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False):
    
    fs = sfreq
    pwd = os.getcwd()

    featurepath = os.getcwd() + '/Features/' + dataset + '/'    
    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']
    electrodeList = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
    print("Number of segments are: {}".format(ans.shape[1]))
    
    #X##############################################################################################
    
    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)
    
    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)

    for k in list(featuresDict.keys()):
        if k not in common:
            # pop out common feature
            featuresDict.pop(k)

    
    
    featuresList = list(featuresDict.keys())
    
    # defining column names
    feature_channel_index = []

    for feature in featuresList:
        for i in range(featuresDict[feature].shape[0]):
            feature_channel_index.append(feature + str(i))
    
    #defining feature matrix
    featureMatrix = np.empty((0,ans.shape[1])) #[14*32 + 1,80640]
    for key,value in featuresDict.items():
        featureMatrix = np.append(featureMatrix,value,axis=0)

    
    print("Shape of FeatureMatrix: {}\n".format(featureMatrix.T.shape))
    
    #data-imputation and nan-removal
    featureMatrix = featureMatrix.astype(np.float32)
    
    if np.isnan(featureMatrix).any():
        featureMatrix = np.nan_to_num(featureMatrix,nan=0)

    X = pd.DataFrame(featureMatrix.T)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X.columns = feature_channel_index
    

    #Y#####################################################################

    y = Y_epoch[:,label] #valence
    # y = pd.DataFrame(y)

    ########################################################################
    rmseList = []

    for col in feature_channel_index:
        input_df = pd.DataFrame(X[col])

        X_train, X_test, y_train, y_test = train_test_split(input_df, y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier
        # clf = xgb.XGBClassifier(verbose = 5)
        # X_train = X_train.reshape(1, -1)
        # X_test = X_test.reshape(1, -1)

        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict, squared=False)
        rmseList.append(rmse)

    

    col_df = pd.DataFrame(feature_channel_index)
    rmse_df = pd.DataFrame(rmseList)
    #concat two dataframes for better visualization 
    colRanking = pd.concat([col_df, rmse_df],axis=1)
    colRanking.columns = ['Column','RMSE']  #naming the dataframe columns
    features_result = colRanking.sort_values('RMSE')
    print(features_result)


    N = len(feature_channel_index)
    topNRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]

    for n in range(1, N+1):
        ranking_df = features_result.head(n)
        topncols = ranking_df['Column'].tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X[topncols], y, test_size=0.2, random_state=42)

        # Normalise-scale data 
        # Feature Scaling
        if(scale == True):
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        # Apply classfier
        # clf = xgb.XGBClassifier(verbose = 5)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict, squared=False)
        topNRmseList.append(rmse)


    topcol_df = pd.DataFrame(topNList)
    toprmse_df = pd.DataFrame(topNRmseList)
    #concat two dataframes for better visualization 
    topcolRanking = pd.concat([topcol_df, toprmse_df],axis=1)
    topcolRanking.columns = ['Column','RMSE']  #naming the dataframe columns
    topfeatures_result = topcolRanking
    print(topfeatures_result)
    topfeatures_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "ColumnsRegressionRanking" + str(window) + str(stride) + ".csv")


    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(60, 9)
    plt.xlabel('Top N Columns')
    plt.ylabel('RMSE')
    plt.title("Top N Columns v/s RMSE Plot for Window:{} Stride:{} epoched data by varying N".format(window,stride))
    plt.plot(topfeatures_result.loc[:,"Column"], topfeatures_result.loc[:,"RMSE"])
    plt.tight_layout()
    # plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "topFeatureColumnsRegressionRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    plt.show()
    plt.clf()


'''
        # search_method = "sklearn"
        # best_clf = None
        # if(search_method == "bayes_sk_opt"):

        #     # BayesCV scikit opt
        #     search_space = {"bootstrap": Categorical([True, False]), # values for boostrap can be either True or False
        #     "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
        #     "max_features": Categorical(['auto', 'sqrt','log2']), 
        #     "min_samples_leaf": Integer(2, 10),
        #     "min_samples_split": Integer(2, 10),
        #     "n_estimators": Integer(100, 500)
        #     }

        #     forest_bayes_search = BayesSearchCV(clf, search_space, n_iter=32, cv=5)
        #     print(forest_bayes_search)
        #     print(forest_bayes_search.fit(X_train, y_train))
        #     print("Best Parameters are: ", forest_bayes_search.best_params_)
        #     best_clf = forest_bayes_search.best_estimator_

        # elif(search_method =="random_grid_search"):
        #     print("Random Search followed by GridSearch initiated!\n");
        #     #RandomSearchCV followed by GridSearchCV
        #     random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        #         'max_features': ['auto', 'sqrt','log2'],
        #         'max_depth': [int(x) for x in np.linspace(10, 1000,10)],
        #         'min_samples_split': [2, 5, 10,14],
        #         'min_samples_leaf': [1, 2, 4,6,8],
        #         }
        #     rf_randomcv=RandomizedSearchCV(estimator=clf,param_distributions=random_grid,n_iter=100,cv=5,verbose=2,random_state=100)        
        #     print(rf_randomcv.fit(X_train, y_train))
        #     print("Best Parameters for RandomSearchCV are: ", rf_randomcv.best_params_)
        #     print("RMSE with RandomSearchCV is :",mean_squared_error(y_test, rf_randomcv.best_estimator_.predict(X_test),squared=False));
            
        #     param_grid = {
        #         'max_depth': [rf_randomcv.best_params_['max_depth']],
        #         'max_features': [rf_randomcv.best_params_['max_features']],
        #         'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
        #                             rf_randomcv.best_params_['min_samples_leaf']+2, 
        #                             rf_randomcv.best_params_['min_samples_leaf'] + 4],
        #         'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
        #                             rf_randomcv.best_params_['min_samples_split'] - 1,
        #                             rf_randomcv.best_params_['min_samples_split'], 
        #                             rf_randomcv.best_params_['min_samples_split'] +1,
        #                             rf_randomcv.best_params_['min_samples_split'] + 2],
        #         'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
        #                         rf_randomcv.best_params_['n_estimators'], 
        #                         rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
        #     }

        #     grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10, verbose=5)
        #     grid_search.fit(X_train,y_train)
        #     best_clf = rf_randomcv.best_estimator_
        # elif search_method =="manual_search":
        #     min_rmse = 1000
        #     best_clf = clf
        #     min_params = None
        #     # 2*3*3*3*3
        #     param_grid = {'n_estimators': [50, 100],
        #     'max_features': ['auto'],
        #     'max_depth': [2, 10, 100],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 8],
        #     }

        #     # param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        #     # 'max_features': ['auto', 'sqrt','log2'],
        #     # 'max_depth': [int(x) for x in np.linspace(10, 1000,10)],
        #     # 'min_samples_split': [2, 5, 10],
        #     # 'min_samples_leaf': [1, 2, 4,6,8],
        #     # }
        #     param_grid = ParameterGrid(param_grid)
        #     for params in param_grid:
        #         print("Current Parameters : ", params)
        #         # cuml_model = curfc(max_features=1.0, n_bins=128,
        #         #     split_algo=0, min_samples_leaf=1,
        #         #     min_samples_split=2,
        #         #     n_estimators=40, accuracy_metric='r2',
        #         #     use_experimental_backend=False)
        #         temp_clf = RandomForestRegressor( max_features = params['max_features'], min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'], n_estimators = params['n_estimators'],max_depth = params['max_depth']);
        #         temp_clf.fit(X_train,y_train)
        #         y_predict = temp_clf.predict(X_test)
        #         rmse = mean_squared_error(y_test, y_predict,squared=False)
        #         print("Current RMSE with above params : ", rmse)
        #         if(min_rmse > rmse):
        #             min_rmse = rmse;
        #             best_clf = temp_clf;
        #             min_params = params;

        #     print("Best Params for parameter search are : \n", min_params)
        #     print("window: {}, stide: {}, rmse: {}".format(window,stride,min_rmse))
        #     topRmseList.append(min_rmse)
        # elif search_method == "tpot":
        #     from tpot import TPOTRegressor;
        #     # TPOT setup
        #     GENERATIONS = 5
        #     POP_SIZE = 100
        #     CV = 5
        #     SEED = 42

        #     tpot = TPOTRegressor(
        #     generations=GENERATIONS,
        #     population_size=POP_SIZE,
        #     random_state=SEED,
        #     config_dict="TPOT cuML",
        #     n_jobs=1, # cuML requires n_jobs=1
        #     cv=CV,
        #     verbosity=2,
        #     )

        #     tpot.fit(X_train, y_train)

        #     y_predict = tpot.predict(X_test)
        #     rmse = mean_squared_error(y_test, y_predict,squared=False)
        #     print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        #     topRmseList.append(rmse)


        # else:
        #     best_clf = clf
        #     best_clf.fit(X_train,y_train)

        
        # if search_method != "manual_search" and search_method != "tpot":
        #     y_predict = best_clf.predict(X_test)
        #     rmse = mean_squared_error(y_test, y_predict,squared=False)
        #     print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))
        #     topRmseList.append(rmse)
'''