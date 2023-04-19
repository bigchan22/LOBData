import pandas as pd
import os
from TrainDataGeneration import append_ORD_TM, append_ORD_VOL, append_TM_GP, GetGroupDataFrame, append_STEP5
from DataInfo import header_df, PathDir, EnvPathDir, env_columns, groupmin, envdataoption, envdatasubfix

if envdataoption not in ['last', 'mean']:
    raise ValueError
if not os.path.isdir(EnvPathDir):
    os.mkdir(EnvPathDir)
    print("Making Env files")
else:
    print("Already made")
FileList = os.listdir(PathDir)
FileList.sort()
FileList = [file for file in FileList if ".csv" in file]
for filename in FileList:
    Data = pd.read_csv(os.path.join(PathDir, filename), names=header_df)
    Data = append_ORD_TM(Data)
    Data = append_ORD_VOL(Data)
    Data = append_TM_GP(Data, groupmin=groupmin)
    Data = append_STEP5(Data)
    env_columns = env_columns
    groupcolumns = ['ORD_DD', 'ISU_CD', 'TM_GP']
    sumcolumns = []
    if envdataoption == "last":
        meancolumns = []
        lastcolumns = env_columns
        savename = filename[:-4] + "_" + envdatasubfix + '.csv'
    elif envdataoption == "mean":
        meancolumns = env_columns
        lastcolumns = []
        savename = filename[:-4] + "_" + envdatasubfix + '.csv'
    else:
        raise ValueError
    GDF = GetGroupDataFrame(Data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
    GDF.set_index(['ISU_CD', 'TM_GP'], inplace=True)
    GDF = GDF.reindex(pd.MultiIndex.from_product(GDF.index.levels))
    GDF = GDF.ffill()
    GDF = GDF.reset_index()

    if os.path.isfile(EnvPathDir + savename):
        "Code already run"
        break
    print(EnvPathDir + savename)
    GDF.to_csv(EnvPathDir + savename, index=False)
print("Env files Made")
