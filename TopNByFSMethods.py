from ImportUtils import *

from sklearn.ensemble import RandomForestRegressor as sklearnrfi
from sklearn.feature_selection import VarianceThreshold
# # import EEGExtract as eeg

# from sklearn.model_selection import train_test_split
# # from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# # import xgboost as xgb
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# # from sklearn.ensemble import RandomForestRegressor

# import os
# import glob
# from scipy import io,signal
# import numpy as np
# import pandas as pd
# from sklearn import preprocessing
# import pickle
# from sklearn.metrics import mean_squared_error
# from sklearn.impute import SimpleImputer


# import matplotlib.pyplot as plt
# # %matplotlib inline
# import seaborn as sns
# import copy
# from sklearn import feature_selection
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier


# def loadFeaturesDict(dataset):
#     featuresDict = {'shannonEntropy': None,
#                 'ShannonRes_delta':None,
#                 'ShannonRes_theta':None,
#                 'ShannonRes_alpha':None,
#                 'ShannonRes_beta':None,
#                 'ShannonRes_gamma':None,
#                 'HjorthComp':None,
#                 'HjorthMob':None,
#                 'falseNearestNeighbor':None,
#                 'medianFreq':None,
#                 'bandPwr_delta':None, 
#                 'bandPwr_theta':None, 
#                 'bandPwr_alpha':None, 
#                 'bandPwr_beta':None, 
#                 'bandPwr_gamma':None,
#                 'stdDev':None,
#                 'diffuseSlowing':None,
#                 'spikeNum':None,
#                 'deltaBurstAfterSpike':None,
#                 'shortSpikeNum':None,
#                 'numBursts':None,
#                 'burstLenMean':None,
#                 'burstLenStd':None,
#                 'numSuppressions':None,
#                 'suppLenMean':None,
#                 'suppLenStd':None
#                 }

#     featurepath = os.getcwd() + '/' + dataset + '/data_extracted/featuresDict/'

#     featuresDict['shannonEntropy'] = np.load(featurepath + "shannonEntropy_1_1.npz", allow_pickle=True)['features']

#     featuresDict['ShannonRes_delta'] = np.load(featurepath + "ShannonRes_sub_bands_delta_1_1.npz", allow_pickle=True)['features']

#     featuresDict['ShannonRes_theta'] = np.load(featurepath + "ShannonRes_sub_bands_theta_1_1.npz", allow_pickle=True)['features']

#     featuresDict['ShannonRes_alpha'] = np.load(featurepath + "ShannonRes_sub_bands_alpha_1_1.npz", allow_pickle=True)['features']

#     featuresDict['ShannonRes_beta'] = np.load(featurepath + "ShannonRes_sub_bands_beta_1_1.npz", allow_pickle=True)['features']

#     featuresDict['ShannonRes_gamma'] = np.load(featurepath + "ShannonRes_sub_bands_gamma_1_1.npz", allow_pickle=True)['features']

#     #lyapunov
#     # featuresDict['hFD'] = np.load(featurepath + d/DEAP_hFD.npy",allow_pickle=True)['features']

#     featuresDict['HjorthComp'] = np.load(featurepath + "Hjorth_complexity_1_1.npz", allow_pickle=True)['features']

#     featuresDict['HjorthMob'] = np.load(featurepath + "Hjorth_mobilty_1_1.npz",allow_pickle=True)['features']

#     featuresDict['falseNearestNeighbor'] = np.load(featurepath + "falseNearestNeighbor_1_1.npz",allow_pickle=True)['features']

#     featuresDict['medianFreq'] = np.load(featurepath + "medianFreq_1_1.npz",allow_pickle=True)['features']

#     featuresDict['bandPwr_delta']=np.load(featurepath+"bandPwr_delta_1_1.npz", allow_pickle = True)['features']

#     featuresDict['bandPwr_theta']=np.load(featurepath + "bandPwr_theta_1_1.npz", allow_pickle = True)['features']
    
#     featuresDict['bandPwr_alpha']=np.load(featurepath + "bandPwr_alpha_1_1.npz", allow_pickle = True)['features']
    
#     featuresDict['bandPwr_beta']=np.load(featurepath + "bandPwr_beta_1_1.npz", allow_pickle = True)['features']

#     featuresDict['bandPwr_gamma']=np.load(featurepath + "bandPwr_gamma_1_1.npz", allow_pickle = True)['features']

#     featuresDict['stdDev'] = np.load(featurepath + "stdDev_1_1.npz",allow_pickle=True)['features']

#     # featuresDict['regularity'] = np.load(featurepath + "",allow_pickle=True)['features']

#     # featuresDict['volt05'] = np.load(featurepath + "",allow_pickle=True)['features']
#     # featuresDict['volt10'] = np.load(featurepath + "",allow_pickle=True)['features']
#     # featuresDict['volt20'] = np.load(featurepath + "",allow_pickle=True)['features']


#     featuresDict['diffuseSlowing'] = np.load(featurepath + "diffuseSlowing_1_1.npz",allow_pickle=True)['features']

#     featuresDict['spikeNum'] = np.load(featurepath + "spikeNum_1_1.npz",allow_pickle=True)['features']

#     featuresDict['deltaBurstAfterSpike'] = np.load(featurepath + "deltaBurstAfterSpike_1_1.npz",allow_pickle=True)['features']

#     featuresDict['shortSpikeNum'] = np.load(featurepath + "shortSpikeNum_1_1.npz", allow_pickle=True)['features']

#     featuresDict['numBursts'] = np.load(featurepath + "numBursts_1_1.npz",allow_pickle=True)['features']

#     featuresDict['burstLenMean'] = np.load(featurepath + "burstLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

#     featuresDict['burstLenStd'] = np.load(featurepath + "burstLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']

#     # featuresDict['burstBandPowers'] = np.load(featurepath + "",allow_pickle=True)['features']

#     featuresDict['numSuppressions'] = np.load(featurepath + "numSuppressions_1_1.npz",allow_pickle=True)['features']

#     featuresDict['suppLenMean'] = np.load(featurepath + "suppressionLen_u_and_sigma_mean_1_1.npz",allow_pickle=True)['features']

#     featuresDict['suppLenStd'] = np.load(featurepath + "suppressionLen_u_and_sigma_std_1_1.npz",allow_pickle=True)['features']

#     return featuresDict
    

    
    

def topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest', ht = False):
    pwd = os.getcwd()
    fs = sfreq
    electrodeList = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    featurepath = os.getcwd() + '/Features/' + dataset + '/' #change
    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']
   
    # print("Number of segments are: {}".format(ans.shape[1]))
    #change
    output_file = os.getcwd() + '/Output/' + dataset + '/' + "EFS_{}_{}_{}_{}.csv".format(type(clf).__name__,label,method, "ht" if ht else "")
    with open(output_file,'a') as fd:
        fd.write("Dataset used: {}\n".format(dataset))
        fd.write("ML Model used: {}\n".format(clf))
        fd.write("Number of segments are: {}\n".format(ans.shape[1]))
    
    featuresDict = None
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


    # featuresToAvoid = ['volt05', 'volt10', 'volt20', 'burstBandPowers','hFD']
    featuresList = list(featuresDict.keys())
    # print(featuresList) #change
    with open(output_file,'a') as fd:
        fd.write("Features List:\n")
        writer = csv.writer(fd)
        writer.writerow(featuresList)
    
    
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
    
    
    print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

    X = pd.DataFrame(featureMatrix.T)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X.columns = feature_channel_index
    
    #################################################################
    y = copy.deepcopy(Y_epoch[:,label]) #valence
    print("y.shape: ", y.shape)
    

    dfscores = None

    if(method == 'RandomForest'):
        '''Random Forest Feature Importances'''
        # estimator = RandomForestRegressor()
        estimator = sklearnrfi()
        fit = estimator.fit(X,y)
        dfscores = pd.DataFrame(fit.feature_importances_)
    elif(method == 'RFE'):
        ''' RFE'''
        selector = RFE(clf, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)
        dfscores = pd.DataFrame(selector.ranking_)

    elif(method == 'SelectKBest'):
        """SelecKBest"""
        #apply SelectKBest class to extract top 10 best features
        func = None
        if mutual_info == False:
            func = f_classif
        else:
            func = mutual_info_classif

        bestfeatures = SelectKBest(score_func=func, k=X.shape[1])
        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)

    
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    features_result = featureScores.nlargest(X.shape[1],'Score')
    print(features_result)
    # uncomment 
    # features_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "CommonElectrodeFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")
    # change
    with open(output_file,'a') as fd:
        fd.write("Electrode Ranking \n\n")
        
    features_result.to_csv(output_file, mode='a', header=True)

    ###################################################################
    topcolumns = features_result['Specs'].values
    topfeatures = []
    topelectrodes = []

    for col in topcolumns:
        feature = col[:-2]
        electrode = int(col[-2:])
        if(feature not in topfeatures):
            topfeatures.append(feature)
        
        if(electrode not in topelectrodes):
            topelectrodes.append(electrode)
    
    ##################################################################################
    
    N =  len(topelectrodes)
    topRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]

    
    for n in range(1,N+1):
        
        electrode_index = topelectrodes[:n]
        print(topelectrodes)
        print(electrode_index)
        # X-Values
        featureMatrix = np.empty((len(featuresList)*len(electrode_index),ans.shape[1]))

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

    
        feature_channel_index = []
        for index in electrode_index:
            for feature in featuresList:
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
    print("Top N Electrodes Cumulative Ranking",topNElectrodeRanking)
    
    with open(output_file,'a') as fd:
        fd.write("Top N Electrodes Ranking\n")

    topNElectrodeRanking.to_csv(output_file, mode='a', header=False)
    # uncomment
    # topNElectrodeRanking.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topCommonElectrodeFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")
    # return features_result
    
    
    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('Top N Electrodes')
    plt.ylabel('RMSE')
    # plt.title("Top N Electrodes v/s RMSE Plot for Window:{} Stride:{} epoched data by varying".format(window,stride))
    plt.plot(topNElectrodeRanking.loc[:,"Electrode"], topNElectrodeRanking.loc[:,"RMSE"])
    plt.tight_layout()
    # plt.savefig(pwd + "/" + dataset + "/plots/" + "topElectrodeFSRegressionRanking"+ method + str(window) + str(stride) + ".svg", bbox_inches='tight')
    # plt.show()
    # plt.clf()

    


  

def topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest', ht = False):

    
    pwd = os.getcwd()
    fs = sfreq
    featurepath = os.getcwd() + '/Features/' + dataset + '/'

    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']
    
    print("Number of segments are: {}".format(ans.shape[1]))

    output_file = os.getcwd() + '/Output/' + dataset + '/' + "FFS_{}_{}_{}_{}.csv".format(type(clf).__name__,label,method, "ht" if ht else "")
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

    
    ##################################################################
    
    featuresList = list(featuresDict.keys())
    print(featuresList)
    with open(output_file,'a') as fd:
        fd.write("Features List:\n")
        writer = csv.writer(fd)
        writer.writerow(featuresList)
    

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

    #################################################################
    y = copy.deepcopy(Y_epoch[:,label]) #valence
    print("y.shape: ", y.shape)
    
    
    dfscores = None

    if(method == 'RandomForest'):
        '''Random Forest Feature Importances'''
        estimator = sklearnrfi() #RandomForestRegressor()
        fit = estimator.fit(X,y)
        dfscores = pd.DataFrame(fit.feature_importances_)
    elif(method == 'RFE'):
        ''' RFE'''
        selector = RFE(clf, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)
        dfscores = pd.DataFrame(selector.ranking_)

    elif(method == 'SelectKBest'):
        """SelecKBest"""
        #apply SelectKBest class to extract top 10 best features
        func = None
        if mutual_info == False:
            func = f_classif
        else:
            func = mutual_info_classif

        bestfeatures = SelectKBest(score_func=func, k=X.shape[1])
        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)

    
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    features_result = featureScores.nlargest(X.shape[1],'Score')
    print(features_result)
    #uncomment features_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "CommonFeatureFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")

    with open(output_file,'a') as fd:
        fd.write("Feature Ranking \n")
        
    features_result.to_csv(output_file, mode='a', header=True)



    ###################################################################
    topcolumns = features_result['Specs'].values
    # print(topcolumns)
    topfeatures = []
    topelectrodes = []

    for col in topcolumns:
        feature = col[:-2]
        electrode = int(col[-2:])
        if(feature not in topfeatures):
            topfeatures.append(feature)
        
        if(electrode not in topelectrodes):
            topelectrodes.append(electrode)
    
    
    ######################################################################
    # TOP-N-FEATURE-RANKING
    print(topfeatures)
    print(topelectrodes)
    N =  len(topfeatures)
    topNRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]


    
    for n in range(1,N+1):
        
        topnfeatures = topfeatures[:n]
        
        # X-Values################################################

        featureMatrix = np.empty((0,ans.shape[1]))
    
        for feature in topnfeatures:
            featureMatrix = np.append(featureMatrix, featuresDict[feature], axis=0)
        
        featureMatrix = featureMatrix.astype('float64')
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

        X = X.astype(np.float32)
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




    
    topNFeatures_df = pd.DataFrame(topNList)

    topNRmse_df = pd.DataFrame(topNRmseList)

    #concat two dataframes for better visualization 
    topNFeaturesRanking = pd.concat([topNFeatures_df, topNRmse_df],axis=1)
    topNFeaturesRanking.columns = ['Feature','RMSE']  #naming the dataframe columns
    print(topNFeaturesRanking)

    with open(output_file,'a') as fd:
        fd.write("Top N Features Ranking\n")

    topNFeaturesRanking.to_csv(output_file, mode='a', header=False)    
    # uncomment topNFeaturesRanking.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topCommonFeatureFSRegressionRanking"+ method + str(window) + str(stride) + ".csv")
    # return features_result
    
    
    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(25, 10)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('Top N Features')
    plt.ylabel('RMSE')
    # plt.title("Top N Features v/s RMSE Plot for Window:{} Stride:{} epoched data by varying N".format(window,stride))
    plt.plot(topNFeaturesRanking.loc[:,"Feature"], topNFeaturesRanking.loc[:,"RMSE"])
    plt.tight_layout()
    # plt.savefig(pwd + "/" + dataset + "/plots/" + "topFeatureFSRegressionRanking"+ method + str(window) + str(stride) + ".svg", bbox_inches='tight', dpi=300)
    # plt.show()
    # plt.clf()


def topFSColumnsRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, isPCA=False, mutual_info = False, method='SelectKBest', ht = True):
    
    fs = sfreq
    pwd = os.getcwd()

    featurepath = os.getcwd() + '/Features/' + dataset + '/'
    ans = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['features']
    Y_epoch = np.load((featurepath + "shannonEntropy_{}_{}.npz").format(window,stride), allow_pickle=True)['Y']
    print("Number of segments are: {}".format(ans.shape[1]))
    
    output_file = os.getcwd() + '/Output/' + dataset + '/' + "FeaturesColumns_{}_{}_{}_{}.csv".format(type(clf).__name__,label,method, "ht" if ht else "")
    with open(output_file,'a') as fd:
        fd.write("Dataset used: {}\n".format(dataset))
        fd.write("ML Model used: {}\n".format(clf))
        fd.write("Number of segments are: {}\n".format(ans.shape[1]))


    #X##############################################################################################
    
    featuresDict = None
    featuresDict = loadFeaturesDict(dataset)

    common = []
    with open('intersection.pkl', 'rb') as f:
        common = pickle.load(f)
    #pcatest
    # for k in list(featuresDict.keys()):
    #     if k not in common:
    #         # pop out common feature
    #         featuresDict.pop(k)

    print("Number of Features:",len(list(featuresDict.keys())))
 
    featuresList = list(featuresDict.keys())
 
    # defining column names
    
    feature_channel_index = []
    for feature in featuresList:
        for i in range(featuresDict[feature].shape[0]):
            if(i>=10):
                feature_channel_index.append(feature +'_'+  str(i))
            else:
                feature_channel_index.append(feature + '_0' + str(i))

    print(len(list(featuresDict.keys())))
    print("Number of Feature-Columns: {}\n".format(len(feature_channel_index)))

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Standard Scale Data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #PCA
    if(isPCA == True):
        pca = PCA(0.95)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    
    #Hyperparameter Tuning using Grid Search
    y_predict = None
    if ht == True:
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        grid_search=GridSearchCV(estimator=clf,param_grid=params_dict[str(type(clf).__name__)],cv=cv, verbose=5)
        grid_search.fit(X_train,y_train)
        best_clf = grid_search.best_estimator_
        print(best_clf)
        
        with open(output_file,'a') as fd:
            fd.write("Best Classifier After GS: {}\n".format(best_clf))

        y_predict = grid_search.predict(X_test)
    else:
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

    rmse = mean_squared_error(y_test, y_predict,squared=False)
    print("window: {}, stide: {}, rmse: {}".format(window,stride,rmse))

    with open(output_file,'a') as fd:
        fd.write("RMSE after Hyperparameter-Tuning: {}\n".format(rmse))

    return 0
    ########################################################################
    dfscores = None

    if(method == 'RandomForest'):
        '''Random Forest Feature Importances'''
        estimator = sklearnrfi() #RandomForestRegressor()
        fit = estimator.fit(X,y)
        dfscores = pd.DataFrame(fit.feature_importances_)
    elif(method == 'RFE'):
        ''' RFE'''
        selector = RFE(clf, n_features_to_select=X.shape[1], step=1)
        selector = selector.fit(X, y)
        dfscores = pd.DataFrame(selector.ranking_)

    elif(method == 'SelectKBest'):
        """SelecKBest"""
        #apply SelectKBest class to extract top 10 best features
        func = None
        if mutual_info == False:
            func = f_classif
        else:
            func = mutual_info_classif

        bestfeatures = SelectKBest(score_func=func, k=X.shape[1])
        fit = bestfeatures.fit(X,y)

        dfscores = pd.DataFrame(fit.scores_)
        



    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Column','Score']  #naming the dataframe columns
    features_result = featureScores.nlargest(X.shape[1],'Score')
    print(features_result)

    N = len(feature_channel_index)
    topNRmseList = []
    topNList = ["{}".format(x) for x in range(1,N+1)]

    for n in range(1, N+1):
        ranking_df = features_result.head(n)
        topncols = ranking_df['Column'].tolist()
        
        input_df = pd.DataFrame(X[topncols])

        X_train, X_test, y_train, y_test = train_test_split(input_df, y, test_size=0.2, random_state=42)

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
        print(n,rmse)
        topNRmseList.append(rmse)


    topcol_df = pd.DataFrame(topNList)
    toprmse_df = pd.DataFrame(topNRmseList)
    #concat two dataframes for better visualization 
    topcolRanking = pd.concat([topcol_df, toprmse_df],axis=1)
    topcolRanking.columns = ['Column','RMSE']  #naming the dataframe columns
    topfeatures_result = topcolRanking
    print(topfeatures_result)
    #uncomment topfeatures_result.to_csv(pwd + "/" + dataset + "/arousal_plots/" + "topFSColumnsRegressionRanking"+method + str(window) + str(stride) + ".csv")
    with open(output_file,'a') as fd:
        fd.write("Top Electrode-Feature Ranking\n")
        topfeatures_result.to_csv(output_file, mode='a', header=False)    

    # Plotting
    fig = plt.gcf()
    fig.set_size_inches(60, 9)

    plt.xlabel('Top N Columns')
    plt.ylabel('RMSE')
    plt.title("Top N Columns v/s RMSE Plot for Window:{} Stride:{} epoched data by varying N".format(window,stride))
    plt.plot(topfeatures_result.loc[:,"Column"], topfeatures_result.loc[:,"RMSE"])
    plt.tight_layout()
    # plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "topFSColumnsRegressionRanking"+method + str(window) + str(stride) + ".svg", bbox_inches='tight', dpi=500)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    pass
    