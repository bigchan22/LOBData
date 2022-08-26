import pandas as pd
import os
from source.TrainDataGeneration import append_ORD_TM,append_ORD_VOL,append_TM_GP,GetGroupDataFrame,append_STEP5
from DataInfo import ISU_list_MBRN_sum_top,MBRN_sum_top,header_df
from datetime import datetime
path_dir = '/Data/ksqord/'
a=os.listdir(path_dir)
a.sort()
columns=header_df+['ORD_TM',
       'ORD_VOL', 'TM_GP', '매도5단계호가합계잔량', '매수5단계호가합계잔량']
groupcolumns=['ORD_DD','ISU_CD','TM_GP']
sumcolumns=[]
meancolumns=['ORD_ACPT_NO', 'REGUL_OFFHR_TP_CD', 'BLKTRD_TP_CD', '호가장처리가격',
       'ASKBID_TP_CD', 'MODCANCL_TP_CD',  'ORD_COND_CD', 'INVST_TP_CD', 'ASK_STEP1_BSTORD_PRC',
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
       '매수10단계호가합계잔량', 'PT_TP_CD','ORD_TM',
       'ORD_VOL', '매도5단계호가합계잔량', '매수5단계호가합계잔량']
lastcolumns=['ORD_TP_CD']
groupmin=10
for filename in a:
    Data=pd.read_csv(os.path.join(path_dir,filename),names=header_df)
    Data=append_ORD_TM(Data)
    Data=append_ORD_VOL(Data)
    Data=append_TM_GP(Data,groupmin=groupmin)
    Data=append_STEP5(Data)
    GDF=GetGroupDataFrame(Data,groupcolumns,sumcolumns,meancolumns,lastcolumns)
    SaveDirPath='/Data/LOBData/CollectedTickData/'
    now=datetime.now()
    savename="GDF_tick"+"_"+filename+'_'+str(groupmin)+'_'+str(now.month)+'_'+str(now.day)+'.csv'
    print(SaveDirPath+savename)
    GDF.to_csv(SaveDirPath+savename)
    with open(SaveDirPath+'DataInformation.txt', "a+") as f:
        f.write(savename+'\t'+path_dir+'\n')
        f.write(str(groupmin)+'\n')
        for col in groupcolumns:
            f.write(str(col)+'\t')
        f.write('\n')