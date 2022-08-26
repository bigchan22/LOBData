from source.RawDataCollection import ReadDataFile, RawDataCollection, SaveCollectedData
import pandas as pd
import os
from DataInfo import ISU_list_MBRN_sum_top, MBRN_sum_top

raw_path_dir = '/Data/ksqord/'
ISUlist = ISU_list_MBRN_sum_top
mbrnlist = MBRN_sum_top
mbrlist = [a[0] for a in mbrnlist]
brnlist = [a[1] for a in mbrnlist]

raw_file_list = os.listdir(raw_path_dir)
raw_file_list.sort()
col_path_dir = '/Data/LOBData/CollectedRawData/'
for filename in raw_file_list:
    SaveCollectedData(raw_path_dir, [filename], SaveDirPath=col_path_dir, ISU_CD=ISUlist, MBR_NO=mbrlist,
                      BRN_NO=brnlist)

import pandas as pd
import os
from datetime import datetime
from source.TrainDataGeneration import append_ORD_TM, append_ORD_VOL, append_TM_GP, GetGroupDataFrame, append_STEP5
from DataInfo import ISU_list_MBRN_sum_top, MBRN_sum_top

col_path_dir = '/Data/LOBData/CollectedRawData/'
col_file_list = os.listdir(col_path_dir)
col_file_list.sort()
columns = ['Unnamed: 0', 'ISU_CD', 'ORD_DD', 'ORD_ACPT_NO', 'REGUL_OFFHR_TP_CD',
           'BLKTRD_TP_CD', '호가장처리가격', 'ASKBID_TP_CD', 'MODCANCL_TP_CD',
           'ORD_TP_CD', 'ORD_COND_CD', 'INVST_TP_CD', 'ASK_STEP1_BSTORD_PRC',
           'ASK_STEP1_BSTORD_RQTY', 'ASK_STEP2_BSTORD_PRC',
           'ASK_STEP2_BSTORD_RQTY', 'ASK_STEP3_BSTORD_PRC',
           'ASK_STEP3_BSTORD_RQTY', 'ASK_STEP4_BSTORD_PRC',
           'ASK_STEP4_BSTORD_RQTY', 'ASK_STEP5_BSTORD_PRC',
           'ASK_STEP5_BSTORD_RQTY', 'BID_STEP1_BSTORD_PRC',
           'BID_STEP1_BSTORD_RQTY', 'BID_STEP2_BSTORD_PRC',
           'BID_STEP2_BSTORD_RQTY', 'BID_STEP3_BSTORD_PRC',
           'BID_STEP3_BSTORD_RQTY', 'BID_STEP4_BSTORD_PRC',
           'BID_STEP4_BSTORD_RQTY', 'BID_STEP5_BSTORD_PRC',
           'BID_STEP5_BSTORD_RQTY', 'ORD_ACPT_TM', 'ORD_QTY', 'ORD_PRC',
           '호가우선순위번호', 'MBR_NO', 'BRN_NO', 'CNTR_CD', 'TRST_PRINC_TP_CD',
           'FORNINVST_TP_CD', 'ORD_MEDIA_TP_CD', '회원사주문시각', '예상체결가격', '예상체결수량',
           '매도총호가잔량', '매수총호가잔량', '호가체결접수순서번호', '직전체결가격', '누적체결수량', '누적거래대금',
           'AGG_TM', '시가', '고가', '저가', 'ORGN_ORD_ACPT_NO', '시장구분코드', '자동취소처리구분코드',
           'MKTSTAT_TP_CD', '매도10단계호가합계잔량', '매수10단계호가합계잔량', 'PT_TP_CD', 'ORD_TM',
           'ORD_VOL', 'TM_GP', '매도5단계호가합계잔량', '매수5단계호가합계잔량']
groupcolumns = ['ORD_DD', 'ISU_CD', 'TM_GP']
sumcolumns = []
meancolumns = ['호가장처리가격', 'ASK_STEP1_BSTORD_PRC', 'ASK_STEP1_BSTORD_RQTY', 'ASK_STEP2_BSTORD_PRC',
               'ASK_STEP2_BSTORD_RQTY',
               'ASK_STEP3_BSTORD_PRC', 'ASK_STEP3_BSTORD_RQTY', 'ASK_STEP4_BSTORD_PRC', 'ASK_STEP4_BSTORD_RQTY',
               'ASK_STEP5_BSTORD_PRC', 'ASK_STEP5_BSTORD_RQTY', 'BID_STEP1_BSTORD_PRC', 'BID_STEP1_BSTORD_RQTY',
               'BID_STEP2_BSTORD_PRC', 'BID_STEP2_BSTORD_RQTY', 'BID_STEP3_BSTORD_PRC', 'BID_STEP3_BSTORD_RQTY',
               'BID_STEP4_BSTORD_PRC', 'BID_STEP4_BSTORD_RQTY', 'BID_STEP5_BSTORD_PRC', 'BID_STEP5_BSTORD_RQTY',
               '매도5단계호가합계잔량', '매수5단계호가합계잔량', '매도10단계호가합계잔량', '매수10단계호가합계잔량',
               '매도총호가잔량', '매수총호가잔량', 'ORD_PRC', '시가', '고가', '저가',
               '누적체결수량', '누적거래대금', '직전체결가격']
# lastcolumns=[column for column in columns if column not in (sumcolumns+meancolumns+groupcolumns)]
lastcolumns = []

groupmin = 10
filename = raw_file_list[1]
Data1 = pd.read_csv(os.path.join(col_path_dir, filename))
Data = append_ORD_TM(Data1)
Data = append_ORD_VOL(Data)
Data = append_TM_GP(Data, groupmin=groupmin)
Data = append_STEP5(Data)
GDF = GetGroupDataFrame(Data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
for filename in raw_file_list[2:]:
    Data1 = pd.read_csv(os.path.join(col_path_dir, filename))
    Data = append_ORD_TM(Data1)
    Data = append_ORD_VOL(Data)
    Data = append_TM_GP(Data, groupmin=groupmin)
    Data = append_STEP5(Data)
    TGDF = GetGroupDataFrame(Data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
    GDF = pd.concat([GDF, TGDF])

from datetime import datetime

TickDirPath = '/Data/LOBData/CollectedTickData/'
now = datetime.now()
savefilename = "GDF" + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(
    now.minute) + '.csv'
print(TickDirPath + savefilename)
GDF.to_csv(TickDirPath + savefilename)
with open(TickDirPath + 'DataInformation.txt', "a+") as f:
    f.write(savefilename + '\t' + col_path_dir + '\n')
    f.write(str(groupmin) + "MBRN_sum_top" + '\n')
    for col in groupcolumns:
        f.write(str(col) + '\t')
    f.write('\n')

#### TickOrd_data
import pandas as pd
import os
from datetime import datetime
from source.TrainDataGeneration import append_ORD_TM, append_ORD_VOL, append_TM_GP, GetGroupDataFrame, append_STEP5
from DataInfo import ISU_list_MBRN_sum_top, MBRN_sum_top

groupcolumns = ['ORD_DD', 'ISU_CD', 'MBR_NO', 'BRN_NO', 'TM_GP', 'ASKBID_TP_CD', 'MODCANCL_TP_CD']
sumcolumns = ['ORD_QTY', 'ORD_VOL']
meancolumns = ['ORD_PRC']
lastcolumns = []
totcolumns = ['ORD_DD', 'ISU_CD', 'MBR_NO', 'BRN_NO', 'ASKBID_TP_CD', 'MODCANCL_TP_CD', 'ORD_ACPT_TM', 'ORD_QTY',
              'ORD_PRC']

Datapathlist = col_file_list[1:]
GDF = pd.DataFrame(columns=totcolumns)
for datapath in Datapathlist:
    print(datapath)
    Data1 = pd.read_csv('/Data/LOBData/CollectedRawData/' + datapath)
    print(datapath, "loading complete")
    Data1 = Data1[totcolumns]
    Data = append_ORD_TM(Data1)
    Data = append_ORD_VOL(Data)  # 새로 추가
    Data = append_TM_GP(Data, groupmin=groupmin)

    GDF_temp = GetGroupDataFrame(Data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
    GDF = pd.concat([GDF, GDF_temp])
TickDirPath = '/Data/LOBData/CollectedTickData/'
now = datetime.now()
filename = "GDF_ORD" + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(
    now.minute) + '.csv'
print(TickDirPath + filename)
GDF.to_csv(TickDirPath + filename)
with open(TickDirPath + 'DataInformation.txt', "a+") as f:
    f.write("filename:" + filename + '\n')
    f.write("groupmin" + str(groupmin) + '\n')
    f.write('group columns:')
    for col in groupcolumns:
        f.write(str(col) + '\t')
    f.write('\n')
    for datapath in Datapathlist:
        f.write(str(datapath) + '\t')
    f.write('\n')
