import os
import pandas as pd
import csv
def get_result_df(result_path,TrainName):
    result_file_list = os.listdir(result_path)
    result_file_list.sort()
    result_file_list= [file  for file in result_file_list if TrainName in file]
    result_file_list= [file  for file in result_file_list if "result" in file]

    data = []
    for filename in result_file_list:
        filepath=result_path+ filename
        with open( filepath,'r') as f:
            reader = csv.reader(f, delimiter='\t')
            # loop over each row in the data
            for row in reader:
                # add the row to the data list
                data.append(row)
    df = pd.DataFrame(data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    col=df.columns[0]
    # extract the prefix from the column name (e.g. 'Acc' from 'Accuracy')
    df[['Model Type', 'MBR', 'BRN&Train']] = df['Model'].str.split('_', expand=True).iloc[:,2:5]
    # rename the column to remove the prefix
    prefix = col.split(':')[0].lower()
    df.rename(columns={col: prefix}, inplace=True)
    for col in df.columns[1:5]:
        # extract the prefix from the column name (e.g. 'Acc' from 'Accuracy')
        prefix = col.split(':')[0].lower()
        # loop over each row in the column
        for i in range(len(df[col])):
            # extract the float value from the string
            value = df.at[i, col].split(':')[1].strip()
            # try to convert the value to a float
            try:
                df.at[i, col] = float(value)
            # if it can't be converted, set it to NaN
            except ValueError:
                df.at[i, col] = float('NaN')

        # rename the column to remove the prefix
        df.rename(columns={col: prefix}, inplace=True)
    df = df.iloc[:,1:]
    df.sort_values(by='MBR',inplace=True)
    df.set_index(keys=['MBR','Model Type'], inplace=True)
    return df