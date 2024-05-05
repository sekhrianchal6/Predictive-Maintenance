#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def TXT2CSV_TrainFile(path, train_File):
    """ Convert Training File in TXT to CSV Format
    -- Adds the column names
    -- Find max cycle for each engine and transforms it to Remaining cycles to failure 
    (Max cycles of engine - Current cycle of sample)
    path: Full or relative path of the directory
    train_File: Name of the file in .txt format
    """
    
    raw_data = pd.read_csv(path + '/' + train_File, sep = ' ', header = None)    
    raw_data.dropna(axis=1, inplace=True)

    #Rename columns of raw data into meaningful names
    col_names = ['Sensor' + str(i) for i in range(1, len(raw_data.columns)-1)]
    col_names.insert(0,'Cycles')
    col_names.insert(0,'Engine')
    raw_data.columns = col_names

    #Extract maximum number of cycles for each engine and merge it into raw data
    temp_df = raw_data[['Engine', 'Cycles']].groupby('Engine').max()
    raw_data = raw_data.merge(temp_df, how='left', on = 'Engine', suffixes=('', '_y'))

    #Remaining number of cycles before failure is calculated for each sample
    raw_data['Remaining Cycles'] = raw_data['Cycles_y'] - raw_data['Cycles']
    raw_data.drop(['Cycles_y'], axis=1, inplace = True)
    raw_data.to_csv(path + '/' + train_File.replace('.txt', '.csv'), index = False)


# In[3]:


def TXT2CSV_TestFile(path, test_File, rul_File):
    """ Convert Test File in TXT to CSV Format
    -- Adds the column names
    -- Extracts the remaining cycles to failure from RUL file    
    -- Find actual max cycle for each engine, by (Max cycles in test file + Remaining cycle from RUL) 
    -- Calculate remaining cycles for each sample by (Max cycles of engine - Current cycle of sample)
    
    path: Full or relative path of the directory
    test_File: Name of the test dataset with .txt
    rul_File: Name of the rul file with .txt 
    """
    raw_data = pd.read_csv(path + '/' + test_File, sep = ' ', header = None)
    rul_data = pd.read_csv(path + '/' + rul_File, sep = ' ', header = None)    
    raw_data.dropna(axis=1, inplace=True)
    
    #Rename columns of raw data into meaningful names
    col_names = ['Sensor' + str(i) for i in range(1, len(raw_data.columns)-1)]
    col_names.insert(0,'Cycles')
    col_names.insert(0,'Engine')
    raw_data.columns = col_names

    #RUL Data extraction and cleaning
    rul_data.dropna(axis=1, inplace=True)
    rul_data.columns=['Cycles']
    rul_data['Engine'] = np.arange(1, len(rul_data)+1)
    
    #Extract maximum number of cycles for each engine and merge it into rul data
    temp_df = raw_data[['Engine', 'Cycles']].groupby('Engine').max()
    rul_data = temp_df.merge(rul_data, how='left', on = 'Engine', suffixes=('', '_y'))
    rul_data['Total Cycles'] = rul_data['Cycles'] + rul_data['Cycles_y'] 
    
    raw_data = raw_data.merge(rul_data[['Engine', 'Total Cycles']], how='left', on = 'Engine', suffixes=('', '_y'))

    #Remaining number of cycles before failure is calculated for each sample
    raw_data['Remaining Cycles'] = raw_data['Total Cycles'] - raw_data['Cycles']
    raw_data.drop(['Total Cycles'], axis=1, inplace = True)
    
    raw_data.to_csv(path + '/' + test_File.replace('.txt', '.csv'), index = False)


# In[ ]:


# path = 'E:/Nextstep/MLProjects/YTS/PredictiveMaintenance/Datasets/NASAJetEngine/CMaps'
# train_data = 'train_FD004.txt'
# test_File = 'test_FD004.txt'
# rul_File = 'RUL_FD004.txt'
# TXT2CSV_TestFile(path, test_File, rul_File)
# TXT2CSV_TrainFile(path, train_data)

