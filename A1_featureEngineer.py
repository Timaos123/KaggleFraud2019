#coding:utf8

import pandas as pd
import numpy as np
import re

def getRandItem(x,myMean,myStd):
    if np.isnan(x)==True:
        return myStd*np.random.randn()+myMean
    else:
        return x

def getRandSer(mySer):
    myMean=mySer.mean()
    myStd=mySer.std()
    return mySer.apply(lambda x:getRandItem(x,myMean,myStd))

def isOnly01(mySeries):
    '''check out whether a series only consists of 0s and 1s'''
    mySetList=list(set(list(mySeries)))
    if len(mySetList)==2 and 1 in mySetList and 0 in mySetList:
        return True
    return False

def normalizeDf(myDf,indexName="TransactionID",yName="isFraud"):
    '''
    normalize dataframe
    ==============================
    inputs:
    myDf: inputted dataframe
    indexName: column which won't be normalized
    ==============================
    return:
    normalized dataframe
    '''
    keyList=list(myDf.keys())
    keyList.remove(indexName)
    keyList.remove(yName)
    for keyItem in keyList:
        if isOnly01(myDf[keyItem])==True:
            keyList.remove(keyItem)
    for keyItem in keyList:
        mean=myDf[keyItem].mean()
        std=myDf[keyItem].std()
        if std==0:
            print("NAN column:",keyItem)
            myDf.drop(labels=keyItem,inplace=True,axis=1)
        else:
            myDf[keyItem]=myDf[keyItem].apply(lambda x:(x-mean)/std+1)
    return myDf

def float2Str(myDf,reStr=""):
    '''
    transform num-like data into str
    '''
    keyList=list(myDf.keys())
    for keyItem in keyList:
        if len(re.findall(reStr,keyItem))>0:
            myDf[keyItem]=myDf[keyItem].astype(str)
    return myDf

def type2ZO(myDf,keptInfo=1):
    '''
    transform str to 0-1
    =================================
    input: 
    myDf: processed dataframe
    keep: the rate to be kept
    =================================
    return:
    processed dataframe
    '''
    keyList=list(myDf.keys())
    tempDict={}
    for keyItem in keyList:
        if myDf[keyItem].dtype=="O":
            itemList=list(myDf[keyItem])
            itemSetList=list(set(list(myDf[keyItem])))
            numList=[itemList.count(itemNameItem) for itemNameItem in itemSetList]
            itemNumDict=dict(list(zip(itemSetList,numList)))
            itemNumList=[[dictItem,itemNumDict[dictItem]] for dictItem in itemNumDict.keys()]
            if len(itemSetList)>3:
                tmpList=[item[1]/sum(numList) for item in itemNumList]#rate of kept information
                
                print("information kept in",keyItem)
                tmpI=0
                for itemNumDictKeyItem in itemNumDict.keys():
                    print(itemNumDictKeyItem,":",tmpList[tmpI])
                    tmpI+=1
            sortedItemNumList=list(sorted(itemNumList,key=lambda x:x[1],reverse=True))
            
            sortedItemList=[sortedItemNumItem[0] for sortedItemNumItem in sortedItemNumList]
            itemKeyList=[]
            for numI in range(len(sortedItemList)):
                if sum([itemNumDict[sortedItemItem] for sortedItemItem in sortedItemList[:numI]])/sum(list(itemNumDict.values()))<=keptInfo:
                    itemKeyList.append(keyItem+"_"+str(sortedItemList[numI]))
                else:
                    itemKeyList.append(keyItem+"_"+str(sortedItemList[numI]))
                    break

            # [keyItem+"_"+str(sortedItemList[numI]) for numI in range(len(sortedItemList)) if sum([itemNumDict[sortedItemItem] for sortedItemItem in sortedItemList[:numI]])/sum(list(itemNumDict.values()))<=keptInfo]

            tempDf=pd.get_dummies(myDf[keyItem],dummy_na=False,prefix=keyItem)
            tempDf=tempDf.loc[:,itemKeyList]
            myDf=myDf.join(tempDf)
            myDf.drop(labels=keyItem,inplace=True,axis=1)
    return myDf

if __name__=="__main__":
    
    print("loading data ...")
    trainDf=pd.read_csv("data/train_exist.csv")
    
    print("dropping na columns...")
    trainDf.dropna(inplace=True,how="all",axis=1)

    print("transforming class to 0-1 (and maybe decompositing?) ...")

    print("--float 2 string ...")
    trainDf=float2Str(trainDf,reStr="card1")

    print("--classification variable to 0-1 variable ...")
    trainDf=type2ZO(trainDf,keptInfo=0.5)

    print("--drop nan ...")
    for keyItem in trainDf.keys():
        if len(re.findall("nan",keyItem))>0:
            trainDf.drop(labels=keyItem,axis=1,inplace=True)

    print("filling na ...")
    trainDf=trainDf.apply(lambda mySer:getRandSer(mySer))

    # print("normalizing ...")
    # trainDf=normalizeDf(trainDf)

    print("saving data ...")
    trainDf.to_csv("data/train_exist.csv",index=None)

    print("finished!")