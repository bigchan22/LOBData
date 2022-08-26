### 필요 라이브러리
import os
import pandas as pd
import numpy as np
from datetime import datetime
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
def ReadDataFile(filepath):
    print("Reading ",filepath)
    df = pd.read_csv(filepath, sep=',', header=None, encoding="cp949")
    df.columns= header_df
    return df
def RawDataCollection(path_dir,datalist=[],**kwargs):
    if datalist==[]:
        filepath_list = os.listdir(path_dir)
        filepath_list = [path_dir+'/'+dataname for dataname in filepath_list] 
    else:
        filepath_list=[path_dir+'/'+dataname for dataname in datalist]
    filepath_list.sort()
    df= pd.DataFrame(columns=header_df)
    for filepath in filepath_list:
        newdf=ReadDataFile(filepath)
        for key, value in kwargs.items():
            newdf=newdf[newdf[str(key)].isin(value)]  
        df=pd.concat([df, newdf], axis=0)
    return df    
def SaveCollectedData(path_dir,datalist=[],SaveDirPath='/Data/LOBData/CollectedRawData',**kwargs):
    collected_df=RawDataCollection(path_dir,datalist=datalist,**kwargs)
    SaveDirPath=SaveDirPath
    now=datetime.now()
    filename=str(datalist[0])+'.csv'
    print(SaveDirPath+filename)
    collected_df.to_csv(SaveDirPath+filename)
    with open(SaveDirPath+'DataInformation.txt', "a+") as f:
        f.write(filename+'\t'+path_dir+'\t'+str(datalist)+'\n')
        for key, value in kwargs.items():
            f.write(str(key)+'\t'+str(value)+'\n')
    