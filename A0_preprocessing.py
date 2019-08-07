#coding:utf8

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

if __name__=="__main__":
    print("loading data ...")
    idDf=pd.read_csv("data/train_identity.csv")
    trainDf=pd.read_csv("data/train_transaction.csv")

    print("calculating size ...")
    originalSize=trainDf.size
    print("original size:",originalSize)

    idList=list(idDf["TransactionID"])
    existDf=trainDf[trainDf["TransactionID"].isin(idList)==True]
    existSize=existDf.size
    print("exist size:",existSize)
    deltaSize=originalSize-existSize
    print("delta size:",deltaSize)

    print("generalizing training data ...")
    existDf.to_csv("data/train_exist.csv",index=None)

    print("finished !")