import pandas as pd
import numpy as np
import os
import yaml
from TrainDataGeneration import append_ORD_TM, append_ORD_VOL, append_TM_GP, GetGroupDataFrame, append_STEP5

# from DataInfo import header_df, PathDir, EnvPathDir, groupmin, \
#     MBRNlist, envdatasubfix, ISUlist, datasubfix, feat_cols, TrainPathDir, \
#     divdict, loglist, minmaxnormlist, groupcolumns, sumcolumns, meancolumns, lastcolumns
with open('DataInfo.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

header_df = config['header_df']
PathDir = config['PathDir']
EnvPathDir = config['EnvPathDir']
groupmin = config['groupmin']
MBRNlist = config['MBRNlist']
envdatasubfix = config['envdatasubfix']
ISUlist = config['ISUlist']
datasubfix = config['datasubfix']
feat_cols = config['feat_cols']
TrainPathDir = config['TrainPathDir']
divdict = config['divdict']
loglist = config['loglist']
minmaxnormlist = config['minmaxnormlist']
groupcolumns = config['groupcolumns']
sumcolumns = config['sumcolumns']
meancolumns = config['meancolumns']
lastcolumns = config['lastcolumns']

columns = header_df


def process_files(file_list):
    tot_df = pd.DataFrame()
    for filename in file_list:
        data, env_data = read_data_files(filename)
        processed_data = preprocess_data(data, env_data)
        train_df = prepare_train_data(processed_data, env_data)
        tot_df += train_df[feat_cols]

    return tot_df


def read_data_files(filename):
    envfilename = filename[:-4] + "_" + envdatasubfix + ".csv"
    print(filename)
    data = pd.read_csv(os.path.join(PathDir, filename), names=header_df)
    env_data = pd.read_csv(os.path.join(EnvPathDir, envfilename))
    return data, env_data


def preprocess_data(data, env_data):
    ISU_list = [a for a in ISUlist if a in env_data.ISU_CD.unique() and a in data.ISU_CD.unique()]
    if BRN is not None:
        data = data[(data['MBR_NO'] == MBR) & (data['BRN_NO'] == BRN)]
    else:
        data = data[(data['MBR_NO'] == MBR)]
    data = data[data['ISU_CD'].isin(ISU_list)]

    data = append_ORD_TM(data)
    data = append_ORD_VOL(data)
    data = append_TM_GP(data, groupmin=groupmin)
    data['NEW_ASK_QTY'] = data['ORD_QTY'] * (data['ASKBID_TP_CD'] == 1) * (1 == data['MODCANCL_TP_CD'])
    data['CCL_ASK_QTY'] = data['ORD_QTY'] * (data['ASKBID_TP_CD'] == 1) * (3 == data['MODCANCL_TP_CD'])
    data['NEW_BID_QTY'] = data['ORD_QTY'] * (data['ASKBID_TP_CD'] == 2) * (1 == data['MODCANCL_TP_CD'])
    data['CCL_BID_QTY'] = data['ORD_QTY'] * (data['ASKBID_TP_CD'] == 2) * (3 == data['MODCANCL_TP_CD'])
    data['NET_ASK_QTY'] = data['NEW_ASK_QTY'] - data['CCL_ASK_QTY']
    data['NET_BID_QTY'] = data['NEW_BID_QTY'] - data['CCL_BID_QTY']
    data['NET_ORD_QTY'] = data['ORD_QTY'] * (2 * data['ASKBID_TP_CD'] - 3) * (2 - data['MODCANCL_TP_CD'])
    data['NET_ORD_VOL'] = data['NET_ORD_QTY'] * (data['ORD_PRC'] + data['직전체결가격'] * (data['ORD_PRC'] == 0))

    GDF = GetGroupDataFrame(data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
    GDF.set_index(['ORD_DD', 'ISU_CD', 'TM_GP'], inplace=True)
    GDF = GDF.reindex(pd.MultiIndex.from_product([GDF.index.levels[0], GDF.index.levels[1], list(range(-9, 55))]))
    GDF = GDF.fillna(0)
    GDF = GDF.reset_index()
    preprocessed_data = GDF.rename(columns={'level_2': 'TM_GP'})
    return preprocessed_data


def prepare_train_data(processed_data, env_data):
    ISU_list = [a for a in processed_data.ISU_CD.unique() if a in env_data.ISU_CD.unique()]
    EnvData_MBRN = env_data[env_data['ISU_CD'].isin(ISU_list)]
    TrainData = pd.concat(
        [processed_data.set_index(['ORD_DD', 'ISU_CD', 'TM_GP']),
         EnvData_MBRN.set_index(['ORD_DD', 'ISU_CD', 'TM_GP'])],
        axis=1)
    TrainData = TrainData.reset_index()
    TrainData = append_STEP5(TrainData)
    TrainData["NET_ORD_QTY2"] = (TrainData["NET_ORD_QTY"] > 0).replace({True: 1, False: 0}) + (
            TrainData["NET_ORD_QTY"] < 0).replace({True: 0, False: 1})

    Train_df = TrainData[(TrainData['TM_GP'] >= 0) & (TrainData['TM_GP'] < 39)]
    for divcol in divdict:
        Train_df[divdict[divcol]] = Train_df[divdict[divcol]].div(Train_df[divcol], axis=0).values
    Train_df[loglist] = np.log(Train_df[loglist])
    Train_df[minmaxnormlist] = (Train_df[minmaxnormlist] - Train_df[minmaxnormlist].min()) / (
            Train_df[minmaxnormlist].max()
            - Train_df[minmaxnormlist].min())
    return Train_df


def save_data(tot_df, mbr, brn):
    data_subfix = f"{mbr}_{brn}{datasubfix}"
    x_data_name = f'Train_ORD_{data_subfix}.npy'
    y_data_name = f'Train_ORD_Label_{data_subfix}.npy'
    x_data_path = os.path.join(TrainPathDir, x_data_name)
    y_data_path = os.path.join(TrainPathDir, y_data_name)
    np.save(x_data_path, tot_df[feat_cols[:]].values.astype('float64'))
    np.save(y_data_path, tot_df['y'].values)
    print("path:", x_data_path)
    print("path:", y_data_path)


for MBR, BRN in MBRNlist:
    print(MBR, BRN)
    file_list = sorted([file for file in os.listdir(PathDir) if ".csv" in file])
    tot_df = process_files(file_list)
    tot_df['y'] = tot_df['NET_ORD_QTY2'].shift(-1).fillna(1)
    tot_df['y'][38::39] = 1
    save_data(tot_df, MBR, BRN)
