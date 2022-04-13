### 필요 라이브러리
import os
import pandas as pd
import numpy as np
import datetime as dt
import math
import warnings
# save numpy array as npy file
from numpy import asarray
from numpy import save




header_df=['ISU_CD', 'ORD_DD', 'ORD_ACPT_NO', 'REGUL_OFFHR_TP_CD', 'BLKTRD_TP_CD', '호가장처리가격',
       'ASKBID_TP_CD', 'MODCANCL_TP_CD', 'ORD_TP_CD', 'ORD_COND_CD', 'INVST_TP_CD', 'ASK_STEP1_BSTORD_PRC',
       'ASK_STEP1_BSTORD_RQTY', 'ASK_STEP2_BSTORD_PRC','ASK_STEP2_BSTORD_RQTY','ASK_STEP3_BSTORD_PRC',
       'ASK_STEP3_BSTORD_RQTY','ASK_STEP4_BSTORD_PRC', 'ASK_STEP4_BSTORD_RQTY','ASK_STEP5_BSTORD_PRC',
       'ASK_STEP5_BSTORD_RQTY',
       'BID_STEP1_BSTORD_PRC', 'BID_STEP1_BSTORD_RQTY', 'BID_STEP2_BSTORD_PRC', 'BID_STEP2_BSTORD_RQTY',
       'BID_STEP3_BSTORD_PRC', 'BID_STEP3_BSTORD_RQTY','BID_STEP4_BSTORD_PRC', 'BID_STEP4_BSTORD_RQTY',
       'BID_STEP5_BSTORD_PRC', 'BID_STEP5_BSTORD_RQTY', 'ORD_ACPT_TM', 'ORD_QTY', 'ORD_PRC',
       '호가우선순위번호', 'MBR_NO', 'BRN_NO', 'CNTR_CD', 'TRST_PRINC_TP_CD', 'FORNINVST_TP_CD',
       'ORD_MEDIA_TP_CD', '회원사주문시각', '예상체결가격', '예상체결수량', '매도총호가잔량', '매수총호가잔량',
       '호가체결접수순서번호', '직전체결가격', '누적체결수량', '누적거래대금', 'AGG_TM', '시가', '고가', '저가',
       'ORGN_ORD_ACPT_NO', '시장구분코드', '자동취소처리구분코드', 'MKTSTAT_TP_CD', '매도10단계호가합계잔량',
       '매수10단계호가합계잔량', 'PT_TP_CD']
def ReadDataFile(filepath,KorOnly=True):
    print("Reading ",filepath)
    df1 = pd.read_csv(filepath, sep=',', header=None, encoding="cp949")
    df1.columns= header_df
    if KorOnly:
        df1=df1[df1['ISU_CD'].str.contains('KR')] 
    return df1

def GetGroupORD_VOL(filepath,ISUlist=[],KorOnly=True):
    df=ReadDataFile(filepath)
    df['ORD_VOL']=df['ORD_QTY']*df['ORD_PRC']
    GDF=df.groupby(['ISU_CD','ASKBID_TP_CD','MBR_NO','BRN_NO'])
    GDF=GDF['ORD_QTY','ORD_VOL'].sum()
    return GDF
def GetGroupORD_VOL_data(data_dir,ISUlist=[],KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    asktot=0
    bidtot=0
    for idx,filename in enumerate(file_list):
        filepath=data_dir+'/'+filename
        if(idx==0):
            GroupDF=GetGroupORD_VOL(filepath)
        else:
            GroupDF= GroupDF.add(GetGroupORD_VOL(filepath),fill_value=0)
    return GroupDF
def GetLastData(filepath,ISUlist=[],KorOnly=True):
    df=ReadDataFile(filepath)
    GDF=df.groupby(['ORD_DD','ISU_CD'])
    first=GDF['ASK_STEP1_BSTORD_PRC','BID_STEP1_BSTORD_PRC'].nth(0)
    last=GDF['ASK_STEP1_BSTORD_PRC','BID_STEP1_BSTORD_PRC'].nth(-1)
    return first,last
def WriteLastData(filepath,ISUlist=[],KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    for filename in file_list:
        filepath=data_dir+'/'+filename
        print(filepath)
        first,last=GetLastData(filepath)
        WriteStatistics(statpath,filepath,"Closing price",last.mean().mean())
def GetAskBidCountByFilePath(filepath, KorOnly=True):
    print("Reading ",filepath)
    df1 = pd.read_csv(filepath, sep=',', header=None, encoding="cp949")
    df1.columns= header_df
    if KorOnly:
        df1=df1[df1['ISU_CD'].str.contains('KR')]
    dfcount=df1[1:]["ASKBID_TP_CD"].squeeze()
    askcount=dfcount.value_counts()[1]
    bidcount=dfcount.value_counts()[2]
    return askcount,bidcount
def GetNumStockDataByFilePath(filepath, KorOnly=True):
    print("Reading ",filepath)
    df1 = pd.read_csv(filepath, sep=',', header=None, encoding="cp949")
    df1.columns= header_df
    if KorOnly:
        df1=df1[df1['ISU_CD'].str.contains('KR')]
    dfcount=df1[1:]["ISU_CD"].squeeze()
    return len(dfcount.value_counts())
def GetOrdVolumeByFilePath(filepath, KorOnly=True):
    print("Reading ",filepath)
    df1 = pd.read_csv(filepath, sep=',', header=None, encoding="cp949")
    df1.columns= header_df
    if KorOnly:
        df1=df1[df1['ISU_CD'].str.contains('KR')]
    dfcount=df1[1:]["ORD_QTY"].squeeze()*df1[1:]["ORD_PRC"].squeeze()
    dfsum=dfcount.sum()
    print(dfsum)
    return dfsum
def WriteStatistics(statpath,datapath,statname,statistics):
    with open(statpath+'/'+statname+'.txt', "a+") as f:
        f.write(datapath+'\t'+str(statistics)+'\n')
def WriteAskBidData(statpath,data_dir, KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    asktot=0
    bidtot=0
    for filename in file_list:

        filepath=data_dir+'/'+filename
        print(filepath)
        askcount,bidcount=GetAskBidCountByFilePath(filepath, KorOnly=KorOnly)
        WriteStatistics(statpath,filepath,"Ask",askcount)
        WriteStatistics(statpath,filepath,"Bid",bidcount)

def WriteNumStockData(statpath,data_dir, KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    asktot=0
    bidtot=0
    for filename in file_list:
        filepath=data_dir+'/'+filename
        numstock=GetNumStockDataByFilePath(filepath, KorOnly=KorOnly)
        print(numstock)        
        WriteStatistics(statpath,filepath,"NumStock",numstock)
def WriteOrdVolume(statpath,data_dir, KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    asktot=0
    bidtot=0
    for filename in file_list:
        filepath=data_dir+'/'+filename
        numstock=GetOrdVolumeByFilePath(filepath, KorOnly=KorOnly)
        print(numstock)        
        WriteStatistics(statpath,filepath,"OrdVolume",numstock)
def GetAskBidCountByDir(data_dir, KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    asktot=0
    bidtot=0
    for filename in file_list:
        filepath=data_dir+'/'+filename
        askcount,bidcount=GetAskBidCountByFilePath(filepath, KorOnly=KorOnly)
        asktot+=askcount
        bidtot+=bidcount
    return asktot,bidtot        
def GetDataSumByFilePath(filepath,featureset, KorOnly=True):
    print("Reading ",filepath)
    df1 = pd.read_csv(filepath, sep=',', header=None, encoding="cp949")
    df1.columns= header_df
    if KorOnly:
        df1=df1[df1['ISU_CD'].str.contains('KR')]
    featuredata=[]
    for feature in featureset:
        dfcount=df1[1:][feature].sum()
        featuredata.append(dfcount)
    return featuredata
def WriteFeatureData(statpath,data_dir, featureset,KorOnly=True):
    file_list = os.listdir(data_dir)
    filecount=0
    asktot=0
    bidtot=0
    for filename in file_list:
        filepath=data_dir+'/'+filename
        featuredata=GetDataSumByFilePath(filepath,featureset)
        print(featuredata)        
        for idx, feature in enumerate(featureset):
            WriteStatistics(statpath,filepath,feature,featuredata[idx])
