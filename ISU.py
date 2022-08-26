#!/usr/bin/env python
# coding: utf-8

# In[1]:


### 필요 라이브러리
import os
import pandas as pd
import numpy as np
import datetime as dt
import math
import warnings
warnings.filterwarnings(action='ignore')
# save numpy array as npy file
from numpy import asarray
from numpy import save
from DataInfo import *
### 데이터 보여지는 개수 설정
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 300)


# In[2]:


mbrnlist= MBRN_sum_top[:15]


# In[3]:


path_dir = '/Data/ksqord'
file_list=os.listdir(path_dir)
file_list.sort()


# In[4]:


columnlist=['MBR_NO','BRN_NO','ISU_CD','ASKBID_TP_CD','MODCANCL_TP_CD','PT_TP_CD']
featlist=columnlist + ['ORD_QTY','ORD_PRC']
filename = file_list[0]
print("extracting stats from", filename )
df = pd.read_csv(path_dir+'/'+filename, sep=',', names=header_df, encoding="cp949")
df['MBRN'] = list(zip(df['MBR_NO'],df['BRN_NO']))
df = df[df['MBRN'].isin(mbrnlist) ]
df = df[featlist]
df.fillna(0)
df['ORD_VOL'] = df['ORD_QTY'] * df['ORD_PRC']
gdf = df.groupby(columnlist).sum()


# In[5]:


for filename in file_list[1:]:
    print("extracting stats from", filename )
    df = pd.read_csv(path_dir+'/'+filename, sep=',', names=header_df, encoding="cp949")
    df['MBRN'] = list(zip(df['MBR_NO'],df['BRN_NO']))
    df = df[df['MBRN'].isin(mbrnlist) ]
    df = df[featlist]
    df.fillna(0)
    df['ORD_VOL'] = df['ORD_QTY'] * df['ORD_PRC']
    tgdf = df[featlist].groupby(columnlist).sum()
    gdf = gdf.add(tgdf).fillna(gdf).fillna(tgdf)





gdf.to_csv('./StatsForDataSelection/AlmostFullColumn_ISU'+"MBRN_sum_top")

tgdf= gdf.reset_index()
tgdf = tgdf.pivot(index= ['MBR_NO', 'BRN_NO', 'ISU_CD','ASKBID_TP_CD','PT_TP_CD'],columns=['MODCANCL_TP_CD'],values='ORD_VOL')
tgdf = tgdf.rename(columns={1: 'NEW', 2:'EDIT',3:'CANCL'})
tgdf=tgdf.fillna(0)
tgdf['NET']= tgdf['NEW']-tgdf['CANCL']
tgdf.sum(level=[2]).sort_values('NEW',ascending=False).index[:100]

