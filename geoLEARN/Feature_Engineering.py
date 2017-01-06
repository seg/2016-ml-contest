# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:15:54 2016

@author: Antoine Caté
"""
import pandas as pd
import numpy as np

###### Import packages needed for the make_vars functions
from scipy.interpolate import interp1d
import pywt

from skimage.filters.rank import entropy
from skimage.morphology import rectangle
from skimage.util import img_as_ubyte


def make_dwt_vars_cD(wells_df,logs,levels,wavelet):

    wave= pywt.Wavelet(wavelet)
    
    grouped = wells_df.groupby(['Well Name'])
    new_df = pd.DataFrame()
    for key in grouped.groups.keys():
    
        depth = grouped.get_group(key)['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth
        for log in logs:
      
            temp_data = grouped.get_group(key)[log]

            cA_4, cD_4, cD_3, cD_2, cD_1 = pywt.wavedec(temp_data,wave,level=4,mode='symmetric')
            dict_cD_levels = {1:cD_1, 2:cD_2, 3:cD_3, 4:cD_4}
                
            for i in levels:
                new_depth = np.linspace(min(depth),max(depth),len(dict_cD_levels[i]))
                fA = interp1d(new_depth,dict_cD_levels[i],kind='nearest')
                temp_df[log + '_cD_level_' + str(i)] = fA(depth)
    
        new_df = new_df.append(temp_df)
        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_dwt_vars_cA(wells_df,logs,levels,wavelet):

    wave= pywt.Wavelet(wavelet)
    
    grouped = wells_df.groupby(['Well Name'])
    new_df = pd.DataFrame()
    for key in grouped.groups.keys():
    
        depth = grouped.get_group(key)['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth
        for log in logs:
      
            temp_data = grouped.get_group(key)[log]
              
            for i in levels:
                
                    cA_cD = pywt.wavedec(temp_data,wave,level=i,mode='symmetric')
                    cA = cA_cD[0]
                    new_depth = np.linspace(min(depth),max(depth),len(cA))
                    fA = interp1d(new_depth,cA,kind='nearest')
                    temp_df[log + '_cA_level_' + str(i)] = fA(depth)
    
        new_df = new_df.append(temp_df)
        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_entropy_vars(wells_df,logs,l_foots):
    
    new_df = pd.DataFrame()
    grouped = wells_df.groupby(['Well Name'])
    
    for key in grouped.groups.keys():
    
        depth = grouped.get_group(key)['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth
        
        for log in logs:
            temp_data = grouped.get_group(key)[log]
            image = np.vstack((temp_data,temp_data,temp_data))
            image -= np.median(image) 
            image /= np.max(np.abs(image))
            image = img_as_ubyte(image)
            
            for l_foot in l_foots:     
                footprint = rectangle(l_foot,3)
                temp_df[log + '_entropy_foot' + str(l_foot)] = entropy(image,footprint)[0,:]
    
        new_df = new_df.append(temp_df)
    
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_gradient_vars(wells_df,logs,dx_list):
    
    new_df = pd.DataFrame()
    grouped = wells_df.groupby(['Well Name'])
    
    for key in grouped.groups.keys():
    
        depth = grouped.get_group(key)['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth

        for log in logs:

            temp_data = grouped.get_group(key)[log]

            for dx in dx_list:

                temp_df[log + 'gradient_dx' + str(dx)] = np.gradient(temp_data,dx)

        new_df = new_df.append(temp_df) 
    
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_moving_av_vars(wells_df,logs,windows):
    
    new_df = pd.DataFrame()
    grouped = wells_df.groupby(['Well Name'])

    for key in grouped.groups.keys():

        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        
        for log in logs:

            temp_data = grouped.get_group(key)[log]
            
            for window in windows:
                temp_df[log + '_moving_av_' + str(window) + 'ft'] = pd.rolling_mean(arg=temp_data, window=window, min_periods=1, center=True)                
                
        new_df = new_df.append(temp_df) 

        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_moving_std_vars(wells_df,logs,windows):
    new_df = pd.DataFrame()
    grouped = wells_df.groupby(['Well Name'])

    for key in grouped.groups.keys():
        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        
        for log in logs:

            temp_data = grouped.get_group(key)[log]
            
            for window in windows:
                temp_df[log + '_moving_std_' + str(window) + 'ft'] = pd.rolling_std(arg=temp_data, window=window, min_periods=1, center=True)                
                
        new_df = new_df.append(temp_df) 

        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_moving_max_vars(wells_df,logs,windows):
    new_df = pd.DataFrame()
    grouped = wells_df.groupby(['Well Name'])

    for key in grouped.groups.keys():
        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        
        for log in logs:

            temp_data = grouped.get_group(key)[log]
            
            for window in windows:
                temp_df[log + '_moving_max_' + str(window) + 'ft'] = pd.rolling_max(arg=temp_data, window=window, min_periods=1, center=True)                
                
        new_df = new_df.append(temp_df) 

        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_moving_min_vars(wells_df,logs,windows):
    new_df = pd.DataFrame()
    grouped = wells_df.groupby(['Well Name'])

    for key in grouped.groups.keys():
        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        
        for log in logs:

            temp_data = grouped.get_group(key)[log]
            
            for window in windows:
                temp_df[log + '_moving_min_' + str(window) + 'ft'] = pd.rolling_min(arg=temp_data, window=window, min_periods=1, center=True)                
                
        new_df = new_df.append(temp_df) 

        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df
    
def make_rolling_marine_ratio_vars(wells_df, windows):

    grouped = wells_df.groupby(['Well Name'])
    new_var = pd.DataFrame()
    for key in grouped.groups.keys():
    
        depth = grouped.get_group(key)['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth
        NM_M = grouped.get_group(key)['NM_M']
        
        for window in windows:
    
            temp_df['Depth'] = grouped.get_group(key)['Depth']
            temp_df['Well Name'] = [key for _ in range(len(NM_M))]
            temp_df['NM_M'] = grouped.get_group(key)['NM_M']
            #We initialize a new variable
            temp_df['Marine_ratio_' + str(window) + '_centered'] = pd.rolling_mean(arg=temp_df['NM_M'], window=window, min_periods=1, center=True)

        new_var = new_var.append(temp_df)

            
    new_var = new_var.sort_index()
    new_var =new_var.drop(['Well Name', 'Depth','NM_M'],axis=1)
    return new_var
    
def make_distance_to_M_up_vars(wells_df):
    grouped = wells_df.groupby(['Well Name'])
    new_var = pd.DataFrame()

    for key in grouped.groups.keys():

        NM_M = grouped.get_group(key)['NM_M'].values

        #We create a temporary dataframe that we reset for every well
        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        temp_df['Well Name'] = [key for _ in range(len(NM_M))]
        #We initialize a new variable
        dist_mar_up = np.zeros(len(NM_M))

        # A variable counting the interval from the ipper marine deposit
        # We initialize it to -99999 since we do not know what's abpve the first log
        count = -1

        for i in range(len(NM_M)):

            if ((NM_M[i] == 1) & (count>-1)):

                count+=0.5
                dist_mar_up[i] += count

            elif NM_M[i] == 2:

                count=0

            else:
                dist_mar_up[i] = count

        temp_df['dist_M_up'] = dist_mar_up

        # We append each well variable to a larger dataframe
        # We use a dataframe to preserve the index
        new_var = new_var.append(temp_df)
    
    new_var = new_var.sort_index()
    new_var =new_var.drop(['Well Name','Depth'],axis=1)
    return new_var
    
def make_distance_to_M_down_vars(wells_df):
    grouped = wells_df.groupby(['Well Name'])
    new_var = pd.DataFrame()

    for key in grouped.groups.keys():

        NM_M = grouped.get_group(key)['NM_M'].values

        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        temp_df['Well Name'] = [key for _ in range(len(NM_M))]

        dist_mar_down = np.zeros(len(NM_M))
        count = -1

        for i in range(len(NM_M)-1,-1,-1):

            if ((NM_M[i] == 1) & (count>-1)):

                count+=0.5
                dist_mar_down[i] += count

            elif NM_M[i] == 2:            
                count=0

            else:
                dist_mar_down[i] = count

        temp_df['dist_M_down'] = dist_mar_down

        new_var = new_var.append(temp_df)
        
    new_var = new_var.sort_index()
    new_var =new_var.drop(['Well Name','Depth'],axis=1)
    return new_var
    
def make_distance_to_NM_up_vars(wells_df):
    grouped = wells_df.groupby(['Well Name'])
    new_var = pd.DataFrame()

    for key in grouped.groups.keys():

        NM_M = grouped.get_group(key)['NM_M'].values

        #We create a temporary dataframe that we reset for every well
        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        temp_df['Well Name'] = [key for _ in range(len(NM_M))]
        #We initialize a new variable
        dist_mar_up = np.zeros(len(NM_M))

        # A variable counting the interval from the ipper marine deposit
        # We initialize it to -99999 since we do not know what's abpve the first log
        count = -1

        for i in range(len(NM_M)):

            if ((NM_M[i] == 2) & (count>-1)):

                count+=0.5
                dist_mar_up[i] += count

            elif NM_M[i] == 1:

                count=0

            else:
                dist_mar_up[i] = count

        temp_df['dist_NM_up'] = dist_mar_up

        # We append each well variable to a larger dataframe
        # We use a dataframe to preserve the index
        new_var = new_var.append(temp_df)
    
    new_var = new_var.sort_index()
    new_var =new_var.drop(['Well Name','Depth'],axis=1)
    return new_var
    
def make_distance_to_NM_down_vars(wells_df):
    grouped = wells_df.groupby(['Well Name'])
    new_var = pd.DataFrame()

    for key in grouped.groups.keys():

        NM_M = grouped.get_group(key)['NM_M'].values

        temp_df = pd.DataFrame()
        temp_df['Depth'] = grouped.get_group(key)['Depth']
        temp_df['Well Name'] = [key for _ in range(len(NM_M))]

        dist_mar_down = np.zeros(len(NM_M))
        count = -1

        for i in range(len(NM_M)-1,-1,-1):

            if ((NM_M[i] == 2) & (count>-1)):

                count+=0.5
                dist_mar_down[i] += count

            elif NM_M[i] == 1:            
                count=0

            else:
                dist_mar_down[i] = count

        temp_df['dist_NM_down'] = dist_mar_down

        new_var = new_var.append(temp_df)
        
    new_var = new_var.sort_index()
    new_var =new_var.drop(['Well Name','Depth'],axis=1)
    return new_var
    
