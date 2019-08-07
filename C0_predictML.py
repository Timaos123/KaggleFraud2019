#coding:utf8

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import A1_featureEngineer as A1

if __name__=="__main__":

    print("loading data ...")
    testDf=pd.read_csv("data/test_transaction.csv")
    trainDf=pd.read_csv("data/train_exist.csv",nrows=1)
    originalTransactionList=list(testDf["TransactionID"])

    print("transforming data type ...")
    testDf=pd.get_dummies(testDf,dummy_na=True)

    print("dropping na ...")
    testDf.dropna(inplace=True,how="all",axis=1)
    testDf=testDf.apply(lambda mySer:A1.getRandSer(mySer))

    print("aligning the keys ...")
    testKeySet=set(testDf.keys())
    trainKeyList=list(trainDf.keys())
    trainKeyList.remove("isFraud")
    trainKeySet=set(trainKeyList)

    print("--padding training keys ...")
    nullKeySet=trainKeySet-testKeySet
    for nullKeyItem in nullKeySet:
        testDf[nullKeyItem]=np.zeros(testDf.shape[0])
        
    print("--deleting testing keys ...")
    deleteList=list(testKeySet-trainKeySet)
    testDf.drop(labels=deleteList,axis=1,inplace=True)

    print("generalizing inputs ...")
    testArr=np.array(testDf)

    print("loading model ...")
    myModel=joblib.load("model/myModel.model")
    preY=myModel.predict(testArr).tolist()

    print("binding id and result ...")
    idPreDf=pd.DataFrame(list(zip(originalTransactionList,preY)),columns=["TransactionID","isFraud"])
    
    print("generalizing submission file ...")
    idPreDf.to_csv("data/submission.csv",index=None)

    print("finished!")