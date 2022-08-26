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


path_dir = '/Data/ksqord'


# In[3]:


file_list=os.listdir(path_dir)
file_list.sort()


# In[6]:


columnlist=['MBR_NO','BRN_NO','ASKBID_TP_CD','MODCANCL_TP_CD','PT_TP_CD']
featlist=columnlist + ['ORD_QTY','ORD_PRC']
filename = file_list[0]
df = pd.read_csv(path_dir+'/'+filename, sep=',', names=header_df, encoding="cp949")
df = df[featlist]
df['ORD_VOL'] = df['ORD_QTY'] * df['ORD_PRC']
gdf = df.groupby(columnlist).sum()
for filename in file_list[1:]:
    print("extracting stats from", filename )
    df = pd.read_csv(path_dir+'/'+filename, sep=',', names=header_df, encoding="cp949")
    df = df[featlist]
    df['ORD_VOL'] = df['ORD_QTY'] * df['ORD_PRC']
    tgdf = df[featlist].groupby(columnlist).sum()
    gdf = gdf.add(tgdf).fillna(gdf).fillna(tgdf)
 #   print(gdf)


# In[ ]:


gdf.to_csv('./StatsForDataSelection/AlmostFullColumn_sum')


# gdf=pd.read_csv('./StatsForDataSelection/columnlistcount')

# gdf
