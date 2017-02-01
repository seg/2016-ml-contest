# Alan Richardson (Ausar Geophysical)
# 2017/01/31

import numpy as np
import scipy.signal
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import clone
from matplotlib import pyplot as plt
import scipy.optimize
from scipy.optimize import lsq_linear
import fastdtw
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import medfilt, gaussian
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor

eps = 1e-5

def load_data():
    train_data = pd.read_csv('../facies_vectors.csv');
    train_data = train_data[train_data['Well Name'] != 'Recruit F9'].reset_index(drop=True)
    validation_data = pd.read_csv('../validation_data_nofacies.csv')
    return pd.concat([train_data, validation_data]).reset_index(drop=True)

def get_wellnames(data):
    return data['Well Name'].unique()

def get_numwells(data):
    return len(get_wellnames(data))

def set_well_value(data, wellname, colname, val):
    data.loc[data['Well Name']==wellname, colname] = val

def get_well_value(data, wellname, colname):
    return data.loc[data['Well Name']==wellname, colname].values[0]

def make_label_encoders(data, names):
    les=[]
    for name in names:
        le=preprocessing.LabelEncoder()
        le.fit(data[name])
        les.append({'name': name, 'le': le})
    return les

def apply_label_encoders(data, les):
    for le in les:
        data['%sClass' % le['name']]=le['le'].transform(data[le['name']])
        data.drop(le['name'], axis=1, inplace=True)

def make_onehot_encoders(data, names):
    ohes = []
    for name in names:
        ohe=preprocessing.OneHotEncoder()
        ohe.fit(data[name])
        ohes.append({'name': name, 'ohe': ohe})
    return ohes

def apply_onehot_encoders(data, ohes):
    for ohe in ohes:
        ohdata = pd.DataFrame(ohe['ohe'].transform(data[ohe['name']]).toarray())
        data=data.join(ohdata)
        data.drop(ohe['name'],axis=1,inplace=True)
    return data

def make_scalers(data, names, stype='Robust'):
    scalers = []
    for name in names:
        if (stype == 'Robust'):
            scaler = preprocessing.RobustScaler()
        elif (stype == 'Standard'):
            scaler = preprocessing.StandardScaler()
        else:
            raise ValueError('unknown stype')
        scaler.fit(data[name].dropna(axis=0, inplace=False).values.reshape(-1, 1))
        scalers.append({'name': name, 'scaler': scaler})
    return scalers

def apply_scalers(data, scalers):
    for scaler in scalers:
        data.loc[~data[scaler['name']].isnull(), scaler['name']] = scaler['scaler'].transform(data[scaler['name']].dropna(axis=0, inplace=False).values.reshape(-1,1))

def neigh_interp(data):
    odata = load_data()
    wellnames = get_wellnames(data)
    formations = data['FormationClass'].unique()
    distformation=np.load('dtw_distformation_fce.npy')
    distformation[pd.isnull(distformation)]=0.0
    # distformation is upper triangular, so add the transpose to make it full
    distformationf = np.zeros([len(wellnames),len(wellnames),len(formations)])
    for fidx in range(len(formations)):
        distformationf[:,:,fidx] = distformation[:,:,fidx]+distformation[:,:,fidx].T
    # We don't have facies for wells 9 or 10, so we don't want any other well
    # to have these as one of their nearest neighbours
    distformationf[:,9,:]=np.inf
    distformationf[:,10,:]=np.inf
    # We also don't want a well to be its own neighbour
    distformationf[distformationf==0.0]=np.inf
    data['NeighbFacies']=0

    k=8
    clf = KNeighborsClassifier(n_neighbors = k, weights = 'distance', leaf_size = 1, p = 1)
    cols = ['GR', 'ILD_log10', 'PHIND', 'RELPOS', 'NM_MClass', 'RGT']
    for wellidx in range(len(wellnames)):
        for fidx in formations:
            # Find the k 'nearest' (as determined by dtw) wells
            neighb = np.argsort(distformationf[wellidx,:,formations.tolist().index(fidx)])[:k]
            # Find the rows in data for these wells
            neighb_rows = np.array([False]*len(data))
            for nidx in neighb:
                neighb_rows = neighb_rows | (data['Well Name']==wellnames[nidx])
            # Select only those rows with formation 'fidx'
            neighb_rows = neighb_rows & (data['FormationClass']==fidx)
            # Rows for the chosen formation in the current well
            my_rows = (data['Well Name']==wellnames[wellidx]) & (data['FormationClass']==fidx)
            # Fit and predict
            if (np.sum(neighb_rows)>0) & (np.sum(my_rows)>0):
                clf.fit(data.loc[neighb_rows, cols].values, odata.loc[neighb_rows, 'Facies'].values.ravel())
                data.loc[my_rows, 'NeighbFacies'] = clf.predict(data.loc[my_rows, cols].values)

# Start of functions associated with finding RGT

def get_pts_per_well(data):
    npts_in_well = data.groupby('Well Name', sort=False).size().values
    cum_pts = np.append([0],np.cumsum(npts_in_well))
    return npts_in_well, cum_pts

def build_Adtw(data, wells, nwells, npts_in_well, cum_pts, cols):
    formations = data['FormationClass'].unique()
    max_num_pairs = int(nwells * (nwells-1) / 2 * np.max(npts_in_well))
    max_nz_in_row = int(np.max(npts_in_well) * 2)
    max_num_rows = max_num_pairs
    max_num_nonzero = max_num_rows * max_nz_in_row
    dist = np.zeros([len(wells),len(wells)])
    distformation = np.zeros([len(wells),len(wells),len(formations)])
    indices = np.zeros(max_num_nonzero,dtype=int)
    indptr = np.zeros(max_num_rows+1,dtype=int)
    Adata = np.zeros(max_num_nonzero)
    b = np.zeros(max_num_rows)
    bounds = np.ones(len(data))
    nz_rows = 0
    nz_indices = 0

    def add_shift_sum(Adata, indices, nz_indices, i, path, cum_pts, wellidx, idx):
        col0 = cum_pts[wellidx]
        col1 = cum_pts[wellidx] + path[i][idx]
        num_added_indices = col1 - col0 + 1
        indices[nz_indices:nz_indices+num_added_indices] = np.arange(col0, col1+1)
        #1-2*idx so when idx=0 put +1 in Adata, when idx=1 put -1 in Adata
        Adata[nz_indices:nz_indices+num_added_indices] = np.ones(num_added_indices)*(1-2*idx)
        return num_added_indices

    def add_row (Adata, indices, indptr, b, nz_rows, nz_indices, i, path, cum_pts, wellidxs):
        num_added_indices = 0
        indptr[nz_rows] = nz_indices
        for idx in [0,1]:
            num_added_indices = add_shift_sum(Adata, indices, nz_indices, i, path, cum_pts, wellidxs[idx], idx)
            nz_indices = nz_indices + num_added_indices
        b[nz_rows] = 0.0
        return nz_indices

    weightsum = 0.0
    for well1idx in range(nwells-1):
        for well2idx in range(well1idx+1, nwells):
            w1df = data.loc[data['Well Name'] == wells[well1idx], cols + ['FormationClass']]
            w2df = data.loc[data['Well Name'] == wells[well2idx], cols + ['FormationClass']]
            w1formations = w1df['FormationClass'].unique()
            w2formations = w2df['FormationClass'].unique()
            nzcols = []
            path = []
            for col in cols:
                if (np.all(np.isfinite(w1df[col])) & np.all(np.isfinite(w2df[col]))):
                    nzcols.append(col)
            for formation in formations:
                if (formation in w1formations) & (formation in w2formations):
                    w1f = w1df.loc[w1df['FormationClass'] == formation, nzcols]
                    w2f = w2df.loc[w2df['FormationClass'] == formation, nzcols]
                    w1 = np.array(w1f.values)
                    w2 = np.array(w2f.values)
                    dist_tmp, path_tmp = fastdtw.dtw(w1, w2, 2)
                    dist[well1idx,well2idx] += dist_tmp
                    distformation[well1idx,well2idx,formations.tolist().index(formation)] = dist_tmp
                    for pair in path_tmp:
                        idx1 = w1f.index[pair[0]]-w1df.index[0]
                        idx2 = w2f.index[pair[1]]-w2df.index[0]
                        path.append((idx1, idx2))
            bounds[cum_pts[well1idx]] = np.max([bounds[cum_pts[well1idx]], path[0][1]])
            bounds[cum_pts[well2idx]] = np.max([bounds[cum_pts[well2idx]], path[0][0]])
            #NOTE delete
            #np.save('path_%d_%d_fce.npy' % (well1idx, well2idx), path, allow_pickle = False)
            pre_nz_rows = nz_rows
            pre_nz_indices = nz_indices
            added_1=-1
            added_2=-1
            for i in range(len(path)):
                if ((path[i][0] != added_1) & (path[i][1] != added_2)):
                    if ((i > 0) & (i < len(path)-1)):
                        if (((path[i][0] != path[i-1][0]) & (path[i][1] != path[i+1][1])) | ((path[i][0] != path[i+1][0]) & (path[i][1] != path[i-1][1]))):
                            nz_indices = add_row(Adata, indices, indptr, b, nz_rows, nz_indices, i, path, cum_pts, [well1idx, well2idx])
                            nz_rows = nz_rows + 1
                            added_1 = path[i][0]
                            added_2 = path[i][1]
                    elif (i>0):
                        if ((path[i][0] != path[i-1][0]) & (path[i][1] != path[i-1][1])):
                            nz_indices = add_row(Adata, indices, indptr, b, nz_rows, nz_indices, i, path, cum_pts, [well1idx, well2idx])
                            nz_rows = nz_rows + 1
                            added_1 = path[i][0]
                            added_2 = path[i][1]
                    else:
                        if ((path[i][0] != path[i+1][0]) & (path[i][1] != path[i+1][1])):
                            nz_indices = add_row(Adata, indices, indptr, b, nz_rows, nz_indices, i, path, cum_pts, [well1idx, well2idx])
                            nz_rows = nz_rows + 1
                            added_1 = path[i][0]
                            added_2 = path[i][1]
            num_matched_pairs = nz_rows - pre_nz_rows + 1
            p = 2.0
            weight = num_matched_pairs * (num_matched_pairs/dist[well1idx, well2idx])**(2.0/p)
            weightsum = weightsum + weight
            Adata[pre_nz_indices : nz_indices] = Adata[pre_nz_indices : nz_indices] * weight

    Adata[:nz_indices] = Adata[:nz_indices] / weightsum
    indptr[nz_rows] = nz_indices
    indptr = indptr[:nz_rows+1]
    np.save('dtw_dist_fce.npy', dist)
    np.save('dtw_distformation_fce.npy', distformation)
    return Adata, indices, indptr, b, bounds, nz_rows, nz_indices

def create_Ab(Adata, indices, indptr, b, nz_rows, nz_indices):
    Adata = Adata[:nz_indices]
    indices = indices[:nz_indices]
    b = b[:nz_rows]
    A = csr_matrix((Adata, indices, indptr), dtype=float)
    return A, b, Adata, indices

def solve_Ax(A, b, bounds, data, wells, nwells, npts_in_well, cum_pts, reg_start_row, its=1):
    res = lsq_linear(A,b,bounds=(bounds, 100.0*np.ones(len(data))),verbose=2,lsmr_tol='auto',max_iter=its)
    wellnames = data['Well Name'].unique()
    k = 0
    for i, wellname in enumerate(wellnames):
        wl = len(data.loc[data['Well Name'] == wellname])
        rgt = np.cumsum(res.x[k:k+wl])
        data.loc[data['Well Name'] == wellname, 'RGT'] = rgt
        k = k+wl

def find_rgt(data, names, its):
    wellnames = get_wellnames(data)
    numwells = get_numwells(data)
    npts_in_well, cum_pts = get_pts_per_well(data)
    Adata, indices, indptr, b, bounds, dtw_rows, dtw_indices = build_Adtw(data, wellnames, numwells, npts_in_well, cum_pts, names)
    A, b, Adata, indices = create_Ab(Adata, indices, indptr, b, dtw_rows, dtw_indices)
    solve_Ax(A, b, bounds, data, wellnames, numwells, npts_in_well, cum_pts, dtw_rows, its)

# End of RGT functions

# Start of feature engineering functions

def find_dist(data):
    wellnames = get_wellnames(data)
    numwells = get_numwells(data)
    dist = np.load('dtw_dist_fce.npy')
    dist[pd.isnull(dist)]=0.0
    distf = dist + dist.T
    numpairs = int(numwells * (numwells-1) / 2)
    A = np.zeros([numpairs, numwells], dtype=int)
    b = np.zeros(numpairs)
    row = 0
    for well1idx in range(numwells-1):
        for well2idx in range(well1idx+1, numwells):
            A[row, well1idx] = 1
            A[row, well2idx] = -1
            b[row] = distf[well1idx, well2idx]
            row += 1
    dist = lsqr(A,b)
    for well1idx in range(numwells):
        set_well_value(data, wellnames[well1idx], 'X1D', dist[0][well1idx])

def interval_cols(intervals):
    cols = []
    for interval in intervals:
        for metric in ['Depth','RGT']:
            cols.append('%sFromPrev%sChange' % (metric, interval))
            cols.append('%sToNext%sChange' % (metric, interval))
            cols.append('%sToNearest%sChange' % (metric, interval))
            cols.append('FracThrough%s%s' % (metric, interval))
            cols.append('%sSize%s' % (interval, metric))
        cols.append('Next%s' % interval)
        cols.append('Prev%s' % interval)
        cols.append('%sCompaction' % interval)
    return cols

def interval_fe(data, intervals):
    for interval in intervals:
        for metric in ['Depth','RGT']:
            df = data.groupby(['Well Name',interval],sort=False)[metric].min().reset_index()
            df.columns =  ['Well Name',interval,'%sPrev%sChange' % (metric, interval)]
            data = pd.merge(data,df,how='left',on = ['Well Name',interval])

            df = data.groupby(['Well Name',interval],sort=False)[metric].max().reset_index()
            df.columns =  ['Well Name',interval,'Max%sBefore%sChange' % (metric, interval)]
            data = pd.merge(data,df,how='left',on = ['Well Name',interval])

            # Set next change to be prev change of next interval. This will cause 'NaN' at the end of each well, so fill those with the max of the interval
            df = data.groupby(['Well Name',interval],sort=False)['%sPrev%sChange' % (metric, interval)].first().reset_index()
            df['%sNext%sChange' % (metric, interval)] = df['%sPrev%sChange' % (metric, interval)].shift(-1).reset_index(drop=True)
            df.drop('%sPrev%sChange' % (metric, interval),axis=1,inplace=True)
            df = df.groupby(['Well Name',interval],sort=False).first()
            for wellname in df.index.levels[0]:
                df.loc[wellname,df.loc[wellname].index[-1]] = np.nan
            df = df.reset_index()
            data = pd.merge(data,df,how='left',on = ['Well Name', interval])
            data.loc[data['%sNext%sChange' % (metric, interval)].isnull(),'%sNext%sChange' % (metric, interval)] = data.loc[data['%sNext%sChange' % (metric, interval)].isnull(),'Max%sBefore%sChange' % (metric, interval)]

            #IntervalSizeMetric
            data['%sSize%s'%(interval,metric)] = data['%sNext%sChange'%(metric,interval)] - data['%sPrev%sChange'%(metric,interval)]
            #MetricFromPrevIntervalChange
            data['%sFromPrev%sChange' % (metric,interval)] = data[metric] - data['%sPrev%sChange' % (metric,interval)]
            #MetricToNextIntervalChange
            data['%sToNext%sChange' % (metric,interval)] = data['%sNext%sChange' % (metric,interval)] - data[metric]
            #MetricToNearestIntervalChange
            data['%sToNearest%sChange' % (metric,interval)] = data[['%sToNext%sChange' % (metric,interval), '%sFromPrev%sChange' % (metric,interval)]].min(axis=1)
            #FracThroughMetricInterval
            data['FracThrough%s%s' % (metric,interval)] = (data[metric] - data['%sPrev%sChange'%(metric,interval)]) / (data['%sSize%s'%(interval,metric)]+eps)

        #Next/PrevInterval
        intervalClass = data.groupby(['Well Name', interval],sort=False)[interval].first()
        intervalClass.name = 'Shift%s' %interval
        nextIntervalClass = intervalClass.shift(-1).reset_index()
        prevIntervalClass = intervalClass.shift(1).reset_index()
        nextIntervalClass.columns =  ['Well Name',interval,'Next%s'%interval]
        prevIntervalClass.columns =  ['Well Name',interval,'Prev%s'%interval]
        nextIntervalClass.loc[nextIntervalClass['Next%s'%interval].isnull(),'Next%s'%interval] = nextIntervalClass.loc[nextIntervalClass['Next%s'%interval].isnull(),interval]
        prevIntervalClass.loc[prevIntervalClass['Prev%s'%interval].isnull(),'Prev%s'%interval] = prevIntervalClass.loc[prevIntervalClass['Prev%s'%interval].isnull(),interval]
        data = pd.merge(data,nextIntervalClass,how='left',on = ['Well Name', interval])
        data = pd.merge(data,prevIntervalClass,how='left',on = ['Well Name', interval])

        #Compaction
        data['%sCompaction'%interval] = data['%sSizeRGT'%interval] / (data['%sSizeDepth'%interval] + eps)

    return data

def measurement_cols(ms):
    cols = []
    for m in ms:
        cols.append('MedFilt%s' % m)
        cols.append('Diff%s' % m)
        cols.append('Diff2%s' % m)
        cols.append('Sharp%s' % m)
    return cols

def measurement_fe(data, ms):

    dfg = data.groupby('Well Name')

    for m in ms:

        #MedFilt NOTE WINDOW CHOICE
        for name,group in dfg[m]:
            data.loc[data['Well Name']==name,'MedFilt%s'%m] = medfilt(group,15)

        #Diff
        for name,group in dfg[m]:
            data.loc[data['Well Name']==name,'Diff%s'%m] = np.gradient(group)

        #Diff2
        for name,group in dfg['Diff%s'%m]:
            data.loc[data['Well Name']==name,'Diff2%s'%m] = np.gradient(group)

        #Sharp
        data['Sharp%s' %m] = data[m] - data['Diff2%s' % m]

    return data

def interval_measurement_cols(intervals, ms):
    cols = []
    for interval in intervals:
        for m in ms:
            cols.append('Mean%s%s' % (interval, m))
            cols.append('DiffMean%s%s' % (interval, m))
            if (interval != 'Local'):
                cols.append('Std%s%s' % (interval, m))
                cols.append('FracStd%s%s' % (interval, m))
    return cols

def interval_measurement_fe(data, intervals, ms):
    for interval in intervals:
        for m in ms:

            # Get dataframe group and rows
            dfg = None
            def rows(data, name):
                return None
            if (interval == 'Well') | (interval == 'Local'):
                dfg = data.groupby('Well Name')
                def rows(data, name):
                    return data['Well Name']==name
            else:
                dfg = data.groupby(['Well Name', interval])
                def rows(data, name):
                    return (data['Well Name']==name[0]) & (data[interval]==name[1])

            # Compute mean and standard deviation
            if (interval != 'Local'):
                #MeanInterval
                for name,group in dfg[m]:
                    data.loc[rows(data, name),'Mean%s%s'% (interval, m)] = np.mean(group)

                #StdInterval
                for name,group in dfg[m]:
                    data.loc[rows(data, name),'Std%s%s'% (interval, m)] = np.std(group)
            else:
                #MeanLocal NOTE WINDOW CHOICE
                gauss = gaussian(5,1)
                gauss /= np.sum(gauss)
                for name,group in dfg[m]:
                    data.loc[rows(data, name),'MeanLocal%s'%m] = np.convolve(group,gauss,'same')

            #DiffMeanInterval
            data['DiffMean%s%s'% (interval, m)] = data[m] - data['Mean%s%s'% (interval, m)]

            #FracStdInterval
            if (interval != 'Local'):
                data['FracStd%s%s'% (interval, m)] = data['DiffMean%s%s'% (interval, m)] / (data['Std%s%s'% (interval, m)]+eps)

    return data

def basic_feature_engineering(data):
    cols = ['X1D', 'Formation3Depth', 'DepthFromSurf', 'WellFracMarine', 'FormationFracMarine', 'DepthFromSurf_divby_RGT', 'FormationSizeDepth_rel_av', 'FormationSizeRGT_rel_av', 'DiffRGT', 'IGR', 'VShaleClavier']

    # Give unique values to each NM_M interval so they can be distinguished below
    # Very hacky method for doing it...
    nmclasssep = np.zeros(len(data['NM_MClass']))
    nmclasssep[1:] = np.cumsum(np.abs(np.diff(data['NM_MClass'].values)))
    nmclasssep[0] = nmclasssep[1]
    data['NM_MClassSep'] = nmclasssep

    intervals = ['FormationClass', 'NM_MClassSep']
    intervals_measurement = intervals + ['Well', 'Local']
    cols += interval_cols(intervals)

    ms=[u'GR', u'ILD_log10', u'DeltaPHI', u'PHIND', u'RELPOS']
    cols += measurement_cols(ms)
    cols += interval_measurement_cols(intervals_measurement, ms)

    # X1D
    find_dist(data)

    # Formation3Depth
    df = data.loc[data['FormationClass']==3].groupby(['Well Name'],sort=False)['Depth'].min().reset_index()
    df.columns =  ['Well Name','Formation3Depth']
    data = pd.merge(data,df,how='left',on = 'Well Name')

    # DepthFromSurf
    df = data.groupby(['Well Name'],sort=False)['Depth'].min().reset_index()
    df.columns =  ['Well Name','SurfDepth']
    data = pd.merge(data,df,how='left',on = ['Well Name'])
    data['DepthFromSurf'] = data['Depth']-data['SurfDepth']
    data.drop('SurfDepth',axis=1,inplace=True)

    # WellFracMarine
    df = data.groupby(['Well Name'],sort=False)['NM_MClass'].mean().reset_index()
    df.columns =  ['Well Name','WellFracMarine']
    data = pd.merge(data,df,how='left',on = ['Well Name'])

    # FormationFracMarine
    df = data.groupby(['Well Name', 'FormationClass'],sort=False)['NM_MClass'].mean().reset_index()
    df.columns =  ['Well Name','FormationClass','FormationFracMarine']
    data = pd.merge(data,df,how='left',on = ['Well Name', 'FormationClass'])
    
    #DepthFromSurf_divby_RGT
    data['DepthFromSurf_divby_RGT'] = data['DepthFromSurf']/data['RGT']

    #DiffRGT
    wellrgt = data.groupby(['Well Name'],sort=False)['RGT']
    for name,group in wellrgt:
        data.loc[data['Well Name']==name,'DiffRGT'] = np.gradient(group)
    
    # Intervals
    data = interval_fe(data, intervals)

    # Remove useless columns
    cols.remove('NextNM_MClassSep')
    cols.remove('PrevNM_MClassSep')

    # FormationSizeDepth_rel_av
    mss=data.groupby(['Well Name','FormationClass'])['FormationClassSizeDepth'].first().reset_index().groupby('FormationClass').mean().values
    data['FormationSizeDepth_rel_av']=data['FormationClassSizeDepth'].values/mss[data['FormationClass'].values.astype(int)].ravel()
    # FormationSizeRGT_rel_av
    mss=data.groupby(['Well Name','FormationClass'])['FormationClassSizeRGT'].first().reset_index().groupby('FormationClass').mean().values
    data['FormationSizeRGT_rel_av']=data['FormationClassSizeRGT'].values/mss[data['FormationClass'].values.astype(int)].ravel()

    #Measurements
    data = measurement_fe(data, ms)
    data = interval_measurement_fe(data, intervals_measurement, ms)

    #IGR
    data['IGR'] = (data['MedFiltGR']-data['MedFiltGR'].min())/(data['MedFiltGR'].max()-data['MedFiltGR'].min())

    #VShaleClavier
    data['VShaleClavier'] = 1.7 * np.sqrt(3.38 - (data['IGR']+0.7)**2)

    return cols, data
        
def predict_pe_feature_engineering(data):
    cols = []
    intervals = ['Facies']
    cols += interval_cols(intervals)

    ms=[u'GR', u'ILD_log10', u'DeltaPHI', u'PHIND', u'RELPOS']
    cols += interval_measurement_cols(intervals, ms)

    data = interval_fe(data, intervals)
    data = interval_measurement_fe(data, intervals, ms)

    return cols, data

def predict_facies2_feature_engineering(data):
    cols = []
    intervals = ['FormationClass', 'NM_MClassSep', 'Well', 'Local']

    ms=['PE']
    cols += measurement_cols(ms)
    cols += interval_measurement_cols(intervals, ms)

    data = measurement_fe(data, ms)
    data = interval_measurement_fe(data, intervals, ms)

    return cols, data

def make_classifier(data, Xcols, Ycols, rows, clf):
    clf.fit(data.loc[~rows, Xcols], data.loc[~rows, Ycols])
    return clf

def classify(data, clf, Xcols, Ycols, rows):
    data.loc[rows, Ycols] = clf.predict(data.loc[rows, Xcols])

def make_regressor(data, Xcols, Ycols, rows, reg):
    reg.fit(data.loc[~rows, Xcols], data.loc[~rows, Ycols])
    return reg

def regress(data, reg, Xcols, Ycols, rows):
    data.loc[rows, Ycols] = reg.predict(data.loc[rows, Xcols])

#NOTE seeds
def run(solve_rgt=False):
    # Load + preprocessing
    odata = load_data()
    if (solve_rgt):
        data = load_data()
        le = make_label_encoders(data, ['Formation', 'NM_M'])
        apply_label_encoders(data, le)
        scalers = make_scalers(data, ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'RELPOS', 'PE', 'FormationClass', u'NM_MClass', u'Facies'])
        apply_scalers(data, scalers)
        #NOTE Max its
        find_rgt(data, [u'DeltaPHI', u'Facies', u'GR', u'ILD_log10', u'NM_MClass', u'PE', u'PHIND', u'RELPOS'], 1)
    else:
        data = pd.read_csv('dtw_out_fce_14000.csv')
        data.drop(u'Unnamed: 0', axis=1, inplace=True)
    # Reset Facies back to their unscaled values
    data['Facies']=odata['Facies'].values
    scalers = make_scalers(data, ['RGT'], stype='Standard')
    apply_scalers(data, scalers)
    neigh_interp(data)
    cols = ['DeltaPHI', 'GR', 'ILD_log10', 'PHIND', 'RELPOS', 'FormationClass', 'NM_MClass', 'RGT', 'NeighbFacies']
    basic_cols, data = basic_feature_engineering(data)
    cols += basic_cols

    seed1=0
    seed2=0
    seed3=0

    facies_rows_to_predict = data['Facies'].isnull()
    pe_rows_to_predict = data['PE'].isnull()

    clf1 = XGBClassifier(base_score=0.5, colsample_bylevel=0.5, colsample_bytree=0.6, gamma=0.01, learning_rate=0.025, max_delta_step=0, max_depth=2, min_child_weight=7, missing=None, n_estimators=500, nthread=-1, objective='multi:softprob', reg_alpha=2, reg_lambda=20, scale_pos_weight=1, seed=seed1, silent=True, subsample=0.2)

    clf2 = XGBClassifier(base_score=0.5, colsample_bylevel=0.3, colsample_bytree=0.8,
       gamma=0.01, learning_rate=0.05, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=500, nthread=-1,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=seed2, silent=True, subsample=0.5) 

    reg1 = XGBRegressor(base_score=0.5, colsample_bylevel=0.5, colsample_bytree=0.1,
       gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=1,
       min_child_weight=10, missing=None, n_estimators=500, nthread=-1,
       objective='reg:linear', reg_alpha=10, reg_lambda=10,
       scale_pos_weight=1, seed=seed3, silent=True, subsample=0.1)

    # Predict facies #1
    Ycol = 'Facies'
    Xcols = cols
    clf = make_classifier(data, Xcols, Ycol, facies_rows_to_predict, clf1)
    classify(data, clf, Xcols, Ycol, facies_rows_to_predict)
    for wellname in get_wellnames(data):
        wd = data.loc[data['Well Name'] == wellname, 'Facies']
        wd = medfilt(wd, kernel_size=5)
        data.loc[data['Well Name'] == wellname, 'Facies1'] = wd
    cols += ['Facies1']
    
    # Predict PE
    predict_pe_cols, data = predict_pe_feature_engineering(data)
    cols += predict_pe_cols
    Ycol = 'PE'
    Xcols = cols
    reg = make_regressor(data, Xcols, Ycol, pe_rows_to_predict, reg1)
    regress(data, reg, Xcols, Ycol, pe_rows_to_predict)
    cols += ['PE']
    
    # Predict facies #2
    predict_facies2_cols, data = predict_facies2_feature_engineering(data)
    cols += predict_facies2_cols
    Ycol = 'Facies'
    Xcols = cols
    clf = make_classifier(data, Xcols, Ycol, facies_rows_to_predict, clf2)
    classify(data, clf, Xcols, Ycol, facies_rows_to_predict)
    for wellname in get_wellnames(data):
        wd = data.loc[data['Well Name'] == wellname, 'Facies']
        wd = medfilt(wd, kernel_size=7)
        data.loc[data['Well Name'] == wellname, 'Facies'] = wd
            
    data = data.loc[(data['Well Name'] == 'STUART') | (data['Well Name'] == 'CRAWFORD'),['Well Name','Depth','Facies']]
    data.to_csv('ar4_submission3.csv')

if __name__ == "__main__":
    run()
