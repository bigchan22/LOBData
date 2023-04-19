import pandas as pd
import numpy as np
import os
from TrainDataGeneration import append_ORD_TM, append_ORD_VOL, append_TM_GP, GetGroupDataFrame, append_STEP5
from DataInfo import header_df, PathDir, EnvPathDir, groupmin, \
    MBRNlist, envdatasubfix, ISUlist, datasubfix, feat_cols, TrainPathDir, \
    divdict, loglist, minmaxnormlist,groupcolumns, sumcolumns, meancolumns, lastcolumns

columns = header_df

for MBR, BRN in MBRNlist:
    TotDF = pd.DataFrame()
    print(MBR,BRN)
    FileList = os.listdir(PathDir)
    FileList.sort()
    FileList = [file for file in FileList if ".csv" in file]

    for filename in FileList:
        envfilename = filename[:-4] + "_" + envdatasubfix + ".csv"
        print(filename)
        Data = pd.read_csv(os.path.join(PathDir, filename), names=header_df)
        Data = append_ORD_TM(Data)
        Data = append_ORD_VOL(Data)
        Data = append_TM_GP(Data, groupmin=groupmin)
        Data['NET_ORD_QTY'] = Data['ORD_QTY'] * (2 * Data['ASKBID_TP_CD'] - 3) * (2 - Data['MODCANCL_TP_CD'])
        Data['NET_ORD_VOL'] = Data['NET_ORD_QTY'] * (Data['ORD_PRC'] + Data['직전체결가격'] * (Data['ORD_PRC'] == 0))

        EnvData = pd.read_csv(EnvPathDir + envfilename)
        ISU_list = [a for a in ISUlist if a in EnvData.ISU_CD.unique() and a in Data.ISU_CD.unique()]
        if BRN is not None:
            Data_MBRN = Data[(Data['MBR_NO'] == MBR) & (Data['BRN_NO'] == BRN)]
        else:
            Data_MBRN = Data[(Data['MBR_NO'] == MBR)]
        Data_MBRN = Data_MBRN[Data_MBRN['ISU_CD'].isin(ISU_list)]



        GDF = GetGroupDataFrame(Data_MBRN, groupcolumns, sumcolumns, meancolumns, lastcolumns)
        GDF.set_index(['ORD_DD', 'ISU_CD', 'TM_GP'], inplace=True)
        GDF = GDF.reindex(pd.MultiIndex.from_product([GDF.index.levels[0], GDF.index.levels[1], list(range(-9, 55))]))
        GDF = GDF.fillna(0)
        GDF = GDF.reset_index()
        GDF = GDF.rename(columns={'level_2': 'TM_GP'})

        ISU_list = [a for a in GDF.ISU_CD.unique() if a in EnvData.ISU_CD.unique()]
        EnvData_MBRN = EnvData[EnvData['ISU_CD'].isin(ISU_list)]
        TrainData = pd.concat(
            [GDF.set_index(['ORD_DD', 'ISU_CD', 'TM_GP']), EnvData_MBRN.set_index(['ORD_DD', 'ISU_CD', 'TM_GP'])],
            axis=1)
        TrainData = TrainData.reset_index()
        TrainData = append_STEP5(TrainData)
        TrainData['10단계호가합계잔량'] = TrainData['매수10단계호가합계잔량'] + TrainData['매도10단계호가합계잔량']
        TrainData["NET_ORD_QTY2"] = (TrainData["NET_ORD_QTY"] > 0).replace({True: 1, False: 0}) + (
                TrainData["NET_ORD_QTY"] < 0).replace({True: 0, False: 1})

        Train_df = TrainData[(TrainData['TM_GP'] >= 0) & (TrainData['TM_GP'] < 39)]
        for divcol in divdict:
            Train_df[divdict[divcol]] = Train_df[divdict[divcol]].div(Train_df[divcol], axis=0).values
        Train_df[loglist] = np.log(Train_df[loglist])
        Train_df[minmaxnormlist] = (Train_df[minmaxnormlist] - Train_df[minmaxnormlist].min()) / (
                Train_df[minmaxnormlist].max()
                - Train_df[minmaxnormlist].min())
        TotDF = TotDF.append(Train_df[feat_cols])
    TotDF['y'] = TotDF['NET_ORD_QTY2'].shift(-1).fillna(1)
    TotDF['y'][38::39] = 1
    DataSubfix = str(MBR) + '_' + str(BRN) + datasubfix
    XDataname = 'Train_ORD' + '_' + DataSubfix + '.npy'
    YDataname = 'Train_ORD_Label_' + '_' + DataSubfix + '.npy'
    np.save(TrainPathDir + XDataname, TotDF[feat_cols[:]].values.astype('float64'))
    np.save(TrainPathDir + YDataname, TotDF['y'].values)
    print("path:", TrainPathDir + XDataname)
    print("path:", TrainPathDir + YDataname)
