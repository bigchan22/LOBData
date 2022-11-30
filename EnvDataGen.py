import pandas as pd
import os
from source.TrainDataGeneration import append_ORD_TM, append_ORD_VOL, append_TM_GP, GetGroupDataFrame, append_STEP5
from DataInfo import header_df, pathdir, envpathdir, env_columns
import argparse

parser = argparse.ArgumentParser(description='Train Config')

parser.add_argument('--groupmin', type=int)
parser.add_argument('--pathdir', type=str, default=pathdir)
parser.add_argument('--envdir', type=str, default=envpathdir)
parser.add_argument('--option', type=str)

args = parser.parse_args()
groupmin = args.groupmin
PathDir = args.pathdir
EnvPathDir = args.envdir
if args.option not in ['last', 'mean']:
    raise ValueError
option = args.option
if not os.path.isdir(EnvPathDir):
    os.mkdir(EnvPathDir)
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
    if option == "last":
        meancolumns = []
        lastcolumns = env_columns
        savename = filename[:-4] + "_" + str(groupmin) + 'min_last.csv'
    elif option == "mean":
        meancolumns = env_columns
        lastcolumns = []
        savename = filename[:-4] + "_" + str(groupmin) + 'min_mean.csv'
    else:
        raise ValueError
    GDF = GetGroupDataFrame(Data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
    GDF.set_index(['ISU_CD', 'TM_GP'], inplace=True)
    GDF = GDF.reindex(pd.MultiIndex.from_product(GDF.index.levels))
    GDF = GDF.ffill()
    GDF = GDF.reset_index()

    print(EnvPathDir + savename)
    GDF.to_csv(EnvPathDir + savename, index=False)
