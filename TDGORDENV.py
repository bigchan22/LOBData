import pandas as pd
import numpy as np
import os
from datetime import datetime
from source.TrainDataGeneration import append_ORD_TM,append_ORD_VOL,append_TM_GP,GetGroupDataFrame,append_STEP5
from DataInfo import ISU_list_MBRN_sum_top,MBRN_sum_top,header_df,env_columns



ISUlist_init=['KR7071930002',
 'KR7040670002',
 'KR7044180008',
 'KR7052420007',
 'KR7043100007',
 'KR7024810004',
 'KR7036540003',
 'KR7058820002',
 'KR7018700005',
 'KR7043200005']
ISUlist_init=['KR7031860000', 'KR7215600008', 'KR7028040004', 'KR7030270003',
       'KR7065450009', 'KR7068270008', 'KR7039670005', 'KR7043200005',
       'KR7086520004', 'KR7203650007']
MBR, BRN =42,1
groupmin=10
datasubfix="1111Train_08"
PathDir = '/Data/ksqord1516/'
EnvPathDir=PathDir+'EnvData/'
TrainPathDir='/Data/LOBData/TrainData/'


columns = header_df
env_columns = env_columns
ord_columns = ['ISU_CD', 'ORD_DD', 'ORD_ACPT_NO', 'BLKTRD_TP_CD', '호가장처리가격', 'ASKBID_TP_CD', 'MODCANCL_TP_CD',
 'ORD_TP_CD', 'ORD_COND_CD', 'INVST_TP_CD', 'ORD_ACPT_TM', 'ORD_QTY', 'ORD_PRC', '호가우선순위번호', 'MBR_NO', 'BRN_NO',
 'CNTR_CD', 'TRST_PRINC_TP_CD', 'FORNINVST_TP_CD', 'ORD_MEDIA_TP_CD', '예상체결가격', '예상체결수량', '호가체결접수순서번호',
 '누적체결수량', '누적거래대금', 'AGG_TM', 'ORGN_ORD_ACPT_NO', '자동취소처리구분코드', 'PT_TP_CD']
CancelCount=True
feat_cols=['매도5단계호가합계잔량', '매수5단계호가합계잔량', '매도10단계호가합계잔량',
       '매수10단계호가합계잔량', '매도총호가잔량', '매수총호가잔량', '고가', '저가',
           '시가', '직전체결가격','NET_ORD_QTY2']
# feat_cols=['매도총호가잔량', '매수총호가잔량', '고가', '저가', 
#            '직전체결가격','NET_ORD_QTY2']


TotDF=pd.DataFrame()

FileList=os.listdir(PathDir)
FileList.sort()
FileList= [ file for file in FileList if ".csv" in file ]

for filename in FileList:
    envfilename = filename[:-4]+"_10min_last.csv"
    print(filename)
    Data=pd.read_csv(os.path.join(PathDir,filename),names=header_df)
    Data=append_ORD_TM(Data)
    Data=append_ORD_VOL(Data)
    Data=append_TM_GP(Data,groupmin=groupmin)
    Data['NET_ORD_QTY']=Data['ORD_QTY']*(2*Data['ASKBID_TP_CD']-3)*(2-Data['MODCANCL_TP_CD'])
    Data['NET_ORD_VOL']=Data['NET_ORD_QTY']*(Data['ORD_PRC'] + Data['직전체결가격']*(Data['ORD_PRC'] == 0))

    EnvData=pd.read_csv(EnvPathDir+envfilename)
    ISU_list= [a for a in ISUlist_init  if a in EnvData.ISU_CD.unique() and a in Data.ISU_CD.unique() ]
    print(ISU_list)
    print(len(ISU_list))
    if BRN is not None:
        Data_MBRN=Data[(Data['MBR_NO']==MBR) & (Data['BRN_NO']==BRN)]
    else:
        Data_MBRN=Data[(Data['MBR_NO']==MBR)]
    Data_MBRN=Data_MBRN[Data_MBRN['ISU_CD'].isin(ISU_list)]

    groupcolumns=['ORD_DD','ISU_CD','TM_GP']
    sumcolumns=['NET_ORD_QTY','NET_ORD_VOL']

    meancolumns=['ORD_PRC']
    lastcolumns=[]

    GDF=GetGroupDataFrame(Data_MBRN,groupcolumns,sumcolumns,meancolumns,lastcolumns)
    GDF.set_index(['ORD_DD','ISU_CD', 'TM_GP'], inplace=True)
    GDF = GDF.reindex(pd.MultiIndex.from_product([GDF.index.levels[0],GDF.index.levels[1],list(range(-9,55))]))
    GDF = GDF.fillna(0)
    GDF = GDF.reset_index()
    GDF=GDF.rename(columns={'level_2': 'TM_GP'})


    ISU_list= [a for a in GDF.ISU_CD.unique() if a in EnvData.ISU_CD.unique() ]
    EnvData_MBRN=EnvData[EnvData['ISU_CD'].isin(ISU_list)]

    TrainData = pd.concat([GDF.set_index(['ORD_DD','ISU_CD', 'TM_GP']),EnvData_MBRN.set_index(['ORD_DD','ISU_CD', 'TM_GP'])],axis=1)
    TrainData=TrainData.reset_index()
    TrainData=append_STEP5(TrainData)
    TrainData['10단계호가합계잔량']=TrainData['매수10단계호가합계잔량']+TrainData['매도10단계호가합계잔량']
    TrainData["NET_ORD_QTY2"]=(TrainData["NET_ORD_QTY"]>0).replace({True: 1, False: 0})+(TrainData["NET_ORD_QTY"]<0).replace({True: 0, False: 1})

    Train_df=TrainData[(TrainData['TM_GP']>=0) & (TrainData['TM_GP']<39)]



    divdict={}
    loglist=[]
    loglist+=['고가','저가','직전체결가격']
    minmaxnormlist=[]
    minmaxnormlist+=['매도총호가잔량', '매수총호가잔량', '고가', '저가', 
               '직전체결가격']

    for divcol in divdict:
        Train_df[divdict[divcol]]=Train_df[divdict[divcol]].div(Train_df[divcol],axis=0).values
    Train_df[loglist]=np.log(Train_df[loglist])
    Train_df[minmaxnormlist]=(Train_df[minmaxnormlist]-Train_df[minmaxnormlist].min())/(Train_df[minmaxnormlist].max()\
                                                                                        -Train_df[minmaxnormlist].min())
    TotDF=TotDF.append(Train_df[feat_cols])
TotDF['y']=TotDF['NET_ORD_QTY2'].shift(-1).fillna(1)
TotDF['y'][38::39]=1
TrainData[["NET_ORD_QTY","NET_ORD_QTY2"]]
DataSubfix = str(MBR) + '_' + str(BRN) + datasubfix
XDataname = 'Train_ORD' + '_' + DataSubfix + '.npy'
YDataname = 'Train_ORD_Label_' + '_' + DataSubfix + '.npy'
np.save(TrainPathDir+XDataname,TotDF[feat_cols[:]].values.astype('float64'))
np.save(TrainPathDir+YDataname,TotDF['y'].values)
print("path:",TrainPathDir+XDataname)
print("path:",TrainPathDir+YDataname)