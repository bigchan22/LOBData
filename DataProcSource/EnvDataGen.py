import pandas as pd
import os
from TrainDataGeneration import GetGroupDataFrame, preprocess_env_data
# from DataInfo import header_df, PathDir, EnvPathDir, env_columns, groupmin, envdataoption, envdatasubfix
import yaml

with open('DataInfo.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

header_df = config['header_df']
groupmin = config['groupmin']
PathDir = config['PathDir']
EnvPathDir = config['EnvPathDir']
env_columns = config['env_columns']
envdataoption = config['envdataoption']
envdatasubfix = config['envdatasubfix']


def check_env_data_option():
    if envdataoption not in ['last', 'mean']:
        raise ValueError


def check_save_data(filename):
    save_name = filename[:-4] + "_" + envdatasubfix + '.csv'
    save_path = os.path.join(EnvPathDir, save_name)
    print(save_path)
    if os.path.isfile(save_path):
        "Code already run"
        return True
    else:
        return False


def create_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
        print("Making Env files")
    else:
        print("Already made")


def process_files(file_list):
    for filename in file_list:
        if check_save_data(filename):
            return
        data = pd.read_csv(os.path.join(PathDir, filename), names=header_df)
        data = preprocess_env_data(data, groupmin)
        gdf = generate_group_data_frame(data)
        save_data(gdf, filename)


def generate_group_data_frame(data):
    groupcolumns = ['ORD_DD', 'ISU_CD', 'TM_GP']
    sumcolumns = []

    if envdataoption == "last":
        meancolumns = []
        lastcolumns = env_columns
    elif envdataoption == "mean":
        meancolumns = env_columns
        lastcolumns = []
    else:
        raise ValueError

    gdf = GetGroupDataFrame(data, groupcolumns, sumcolumns, meancolumns, lastcolumns)
    gdf.set_index(['ISU_CD', 'TM_GP'], inplace=True)
    gdf = gdf.reindex(pd.MultiIndex.from_product(gdf.index.levels))
    gdf = gdf.ffill()
    gdf = gdf.reset_index()
    return gdf


def save_data(gdf, filename):
    save_name = filename[:-4] + "_" + envdatasubfix + '.csv'
    save_path = os.path.join(EnvPathDir, save_name)
    print(save_path)
    gdf.to_csv(save_path, index=False)


check_env_data_option()
create_directory(EnvPathDir)

file_list = sorted([file for file in os.listdir(PathDir) if ".csv" in file])
process_files(file_list)

print("Env files Made")
