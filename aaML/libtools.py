import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pprint
from mpl_toolkits.axes_grid1 import make_axes_locatable

# turn off ipython warnings 
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def applyPCA(df):
    """
    Receiver a dataset of features

    Return:
        Print of variance_ratio_ in percent
    """

    columns = df.columns
    scaler = StandardScaler()  
    pca = PCA()
     
    scaler.fit(df)
    df = scaler.transform(df)
    
    pca.fit(df)
    x_pca = pca.transform(df)
    p = pca.explained_variance_ratio_
    result = {}
    for key, perc in zip(columns, p):
        result[key] = "{:.2f}%".format(perc*100)
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)

def litologia(facies):
    if facies == 1:
        return 'Nonmarine sandstone'
    if facies == 2:
        return 'Nonmarine coarse siltstone'
    if facies == 3:
        return 'Nonmarine fine siltstone'
    if facies == 4:
        return 'Marine siltstone and shale'
    if facies == 5:
        return 'Mudstone (limestone)'
    if facies == 6:
        return 'Wackestone (limestone)'
    if facies == 7:
        return 'Dolomite'
    if facies == 8:
        return 'Packstone-grainstone (limestone)'
    if facies == 9:
        return 'Phylloid-algal bafflestone (limestone)'

# Function to obtain Density Porosity    
def phi_n(df):
    return (df['DeltaPHI']+2*df['PHIND'])/2

# Function to obtain Density Neutronic
def phi_d(df):
    return (-df['DeltaPHI']+2*df['PHIND'])/2

# Function used by Brendom to put new label in facies
def label_facies(row, labels):
    return labels[ row['Facies'] -1]

# Bredom's function that shows the well logs
def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)

def label_formation(df,formacao):
    list_form = []
    for df_form in df:
        for i in range (0,len(formacao)):
            if df_form == formacao[i]:
                list_form.append(i+1)
    return list_form

def label_two_groups_formation(formacao):
    formacao
    if formacao[:][-2:] == 'SH':
        return 1
    if formacao[:][-2:] == 'LM':
        return 2

def NM_M_TOPO(df):
    
    RELPOS_NM_M = []
    name_well = df['Well Name'].unique()
    
    for well in name_well:
        
        NM_M = df[df['Well Name'] == well]['NM_M']
        NM_total = len(NM_M[NM_M == 1])
        M_total = len(NM_M[NM_M == 2])
        count_NM = 0 ; count_M = 0
    
        for i in NM_M:
            if i == 1:
                aux_NM = 1-(count_NM/(NM_total-1))
                RELPOS_NM_M.append(aux_NM)
                count_NM = count_NM+1
            if i == 2:
                aux_M = 1-(count_M/(M_total-1))
                RELPOS_NM_M.append(aux_M)
                count_M = count_M+1
   
    return RELPOS_NM_M

def SH_LM_TOPO(df):
    
    SH_LM = []
    name_well = df['Well Name'].unique()
    
    for well in name_well:
        
        S_L = df[df['Well Name'] == well]['Label_Form_SH_LM']
        S_total = len(S_L[S_L == 1])
        L_total = len(S_L[S_L == 2])
        count_S = 0 ; count_L = 0
    
        for i in S_L:
            if i == 1:
                aux_S = 1-(count_S/(S_total-1))
                SH_LM.append(aux_S)
                count_S = count_S+1
            if i == 2:
                aux_L = 1-(count_L/(L_total-1))
                SH_LM.append(aux_L)
                count_L = count_L+1
   
    return SH_LM

def divisao_sh(df):
    return df[df['Label_Form_SH_LM']==1].drop(['Label_Form_SH_LM'],axis=1)

def divisao_lm(df):
    return df[df['Label_Form_SH_LM']==2].drop(['Label_Form_SH_LM'],axis=1)

def divisao_nm(df):
    return df[df['NM_M']==1].drop(['NM_M'],axis=1)

def divisao_m(df):
    return df[df['NM_M']==2].drop(['NM_M'],axis=1)

def divisao_topo(df):
    return df[df['RELPOS']>=0.5].drop(['RELPOS'],axis=1)

def divisao_base(df):
    return df[df['RELPOS']<0.5].drop(['RELPOS'],axis=1)

def divisao_topo_sh_lm(df):
    return df[df['RELPOS_SH_LM']>=0.5].drop(['RELPOS_SH_LM'],axis=1)

def divisao_base_sh_lm(df):
    return df[df['RELPOS_SH_LM']<0.5].drop(['RELPOS_SH_LM'],axis=1)

def divisao_topo_nm(df):
    return df[df['RELPOS_NM_M']>=0.5].drop(['RELPOS_NM_M'],axis=1)

def divisao_base_nm(df):
    return df[df['RELPOS_NM_M']<0.5].drop(['RELPOS_NM_M'],axis=1)
