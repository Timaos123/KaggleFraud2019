#coding:utf8

import pandas as pd
import tqdm as tqdm
import os
import re

if __name__=="__main__":
    
    print("see exact data ...")
    selectedDf=pd.read_csv("data/train_exist.csv",nrows=5)
    print(list(selectedDf.keys()))

    # print("data description ...")
    # for fileItem in tqdm.tqdm(os.listdir("data")):
    #     if ".csv" in fileItem:
    #         fileDf=pd.read_csv("data/"+fileItem)
    #         print(fileItem,":",fileDf.shape)

    # print("find unique descrete data ...")
    # for keyItem in selectedDf.keys():
    #     if len(re.findall(keyItem,"card[0-1]{0,1}"))>0:
    #         dupSet=set(list(selectedDf["card1"].duplicated()))
    #         print("train_transaction/",keyItem,":",dupSet)

