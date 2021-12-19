# Import essential libraries and external file modules

from ImportUtils import *
from TopNByFSMethods import *
from TopNByClassifier import *
from EpochedFeatures import *
from args_eeg import args as my_args



if __name__ == '__main__':

    # args object to fetch command line inputs
    args = my_args()
    print(args.__dict__)
    pwd = os.getcwd()

    dataset = args.dataset
    window = args.window
    stride = args.stride
    sfreq = args.sfreq
    model = args.model
    label = args.label 
    approach = args.approach #byclassifier or byfs
    ml_algo = args.ml_algo #classification or regression
    top = args.top #e or f or ef
    fs_method = args.fs_method

    model_LR = LinearRegression()
    model_SVR = SVR()
    model_RFR = RandomForestRegressor()
    model_XGB = XGBRegressor()
    model_KNN = KNeighborsRegressor()
    model_DTR = DecisionTreeRegressor()

    model_dict = {
        "lr" : model_LR,
        "svr" : model_SVR,
        "rfr" : model_RFR,
        "xgb" : model_XGB,
        "knn" : model_KNN,
        "dtr" : model_DTR
    }

    clf = model_dict[model]
    # sys.exit()
    #feature extraction
    #getEpochedFeatures(dataset, window, stride, sfreq, label)

    if(top == "e"):
        if(approach == "byclassifier"):
            topElectrodeRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, ht = True)
        elif approach == "byfs":
            topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method=fs_method, ht = True)
        
    elif(top == "f"):
        if(approach == "byclassifier"):
            topFeaturesRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, ht = True)
        elif approach == "byfs":
            topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method=fs_method, ht = True)
    elif(top == "ef"):
        topFSColumnsRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, isPCA=True, mutual_info = False, method='SelectKBest', ht = True)

    # if(top == "e"):
    #     # topElectrodeRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)
        
    #     topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest')
    #     sys.exit();
    #     topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='RandomForest')
    #     plt.legend(["Method A","Method B", "Method C"])

    #     if(label == 1):
    #         plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "CorrectedElectrodewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    #         # plt.savefig(pwd + "/" + dataset + "/plots/" + "ElectrodewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    #         plt.show()
    #         plt.clf()
        
    #     else:
    #         plt.savefig(pwd + "/" + dataset + "/plots/" + "CorrectedElectrodewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    #         # plt.savefig(pwd + "/" + dataset + "/plots/" + "ElectrodewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    #         plt.show()
    #         plt.clf()    
        
    # elif(top == "f"):
    #     topFeaturesRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)
    #     # sys.exit();
    #     topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='SelectKBest')
    #     topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method='RandomForest')
    #     if(label == 1):
    #         plt.legend(["Method A","Method B", "Method C"])
    #         plt.savefig(pwd + "/" + dataset + "/arousal_plots/" + "CorrectedFeaturewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    #         plt.show()
    #         plt.clf()
    #     else:
    #         plt.legend(["Method A","Method B", "Method C"])
    #         plt.savefig(pwd + "/" + dataset + "/plots/" + "CorrectedFeaturewiseRanking" + str(window) + str(stride) + ".svg", bbox_inches='tight')
    #         plt.show()
    #         plt.clf()


    # if(approach == "byclassifier"):
    #     clf = RandomForestRegressor()
    #     if(top == "e"):
    #         topElectrodeRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)
    #     elif(top == "f"):
    #         topFeaturesRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)
    #     elif(top == "ef"):
    #         topFeatureColumnsRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False)

    # elif(approach == "byfs"):
    #     clf = RandomForestRegressor()
    #     if(top == "e"):
    #         topElectrodeFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method=fs_method)
    #     elif(top == "f"):
    #         topFeatureFSRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method=fs_method)
    #     elif(top == "ef"):
    #         topFSColumnsRegressionRanking(dataset, window, stride, sfreq, clf, label, scale=False, pca=False, mutual_info = False, method=fs_method)
    
     
        



    

