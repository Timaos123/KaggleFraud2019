#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,recall_score,precision_score
from sklearn.externals import joblib

if __name__=="__main__":

    K=4

    print("training on transaction with single model...")

    print("--loading data ...")
    trainTraDf=pd.read_csv("data/train_exist.csv")

    print("--configuring data ...")
    xLabelList=list(trainTraDf.keys())
    xLabelList.remove("isFraud")
    X=np.array(trainTraDf.loc[:,xLabelList])
    y=np.array(trainTraDf["isFraud"])

    print("--splitting data ...")
    kfModel=KFold(n_splits=K,shuffle=True)

    print("--building model ...")
    myGBDT=GradientBoostingClassifier()
    myBagging=BaggingClassifier(SVC(C=0.5),n_estimators=100)

    print("--training model...")
    myROC=0
    rocList=[]
    for k in tqdm.tqdm(kfModel.split(X)):
        trainX=X[k[0]]
        trainY=y[k[0]]
        testX=X[k[1]]
        testY=y[k[1]]

        myGBDT.fit(trainX,trainY)
        preY1=myGBDT.predict(testX)
        
        myBagging.fit(trainX,trainY)
        preY2=myBagging.predict(testX)

        preY=(preY1+preY2)/2

        try:
            tmpROC=roc_auc_score(testY,preY)
        except:
            continue
        rocList.append(tmpROC)
        if tmpROC>myROC:
            myROC=tmpROC
            joblib.dump(myGBDT,"model/myModel.model")
            print("roc:",myROC)
            print("recall:",recall_score(testY,preY))
            print("precision:",precision_score(testY,preY))
    plt.plot(rocList)
    plt.savefig("img/"+str(K)+"-fold.jpg")