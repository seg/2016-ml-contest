from __future__ import print_function
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.preprocessing import RobustScaler

# The functions below come from the official repository of the contest:
# https://github.com/seg/2016-ml-contest

def display_cm(cm, labels, hide_zeros=False,
                             display_metrics=False):
    """Display confusion matrix with labels, along with
       metrics such as Recall, Precision and F1 score.
       Based on Zach Guo's print_cm gist at
       https://gist.github.com/zachguo/10296432
    """

    precision = np.diagonal(cm)/cm.sum(axis=0).astype('float')
    recall = np.diagonal(cm)/cm.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)
    
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0
    
    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_F1 = np.sum(F1 * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    #print total_precision
    
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + " Pred", end=' ')
    for label in labels: 
        print("%{0}s".format(columnwidth) % label, end=' ')
    print("%{0}s".format(columnwidth) % 'Total')
    print("    " + " True")
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)): 
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeros:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            print(cell, end=' ')
        print("%{0}d".format(columnwidth) % sum(cm[i,:]))
        
    if display_metrics:
        print()
        print("Precision", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % precision[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_precision)
        print("   Recall", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % recall[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_recall)
        print("       F1", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % F1[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_F1)
    
                  
def display_adj_cm(
        cm, labels, adjacent_facies, hide_zeros=False, 
        display_metrics=False):
    """This function displays a confusion matrix that counts 
       adjacent facies as correct.
    """
    adj_cm = np.copy(cm)
    
    for i in np.arange(0,cm.shape[0]):
        for j in adjacent_facies[i]:
            adj_cm[i][i] += adj_cm[i][j]
            adj_cm[i][j] = 0.0
        
    display_cm(adj_cm, labels, hide_zeros, 
                             display_metrics)

def make_facies_log_plot(logs, facies_colors, figsize=(8,12)):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=figsize)
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

    return f, ax



def make_facies_plot_side_by_side(log1, log2, facies_colors, figsize=(5,10)):
    ''' '''
    #make sure logs are sorted by depth
    log1 = log1.sort_values(by='Depth')
    log2 = log2.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=log2.Depth.min(); zbot=log2.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(log1['Facies'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(log2['Facies'].values,1), 100, 1)
    
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    im1 = axes[0].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    im2 = axes[1].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    
    axes[1].set_xlabel('Facies')

    f.suptitle('Well: %s (pred:left, truth:right)' % log1.iloc[0]['Well Name'], fontsize=14,y=0.94)

    return f, axes


# Below are some utility functions I wrote to prepare the data to be fed in the convolutional net.
# Warning: These will need to be rewritten in a much cleaner and vectorized way.

def most_common(lst):
    '''Find the most common element in a list. 
    http://stackoverflow.com/a/1518632'''
    return max(set(lst), key=lst.count)


def redundant_to_final(n_samples, y_pred_redundant_dense, y_abs_idx):
    ''' '''
    y_pred_final_ = {}
    for i in range(n_samples):
        y_pred_final_[i] = []
    
    for i in range(y_pred_redundant_dense.shape[0]):
        y_pred_final_[y_abs_idx[i]].append(y_pred_redundant_dense[i])
        
    y_pred_final = np.zeros((n_samples))
    for i in range(y_pred_final.shape[0]):
        y_pred_final[i] = most_common(y_pred_final_[i])
        
    return y_pred_final.astype(int)

def dense_to_one_hot(dense_labels, n_classes):
    '''9 classes from 1 to 9.
    TO DO: vectorize  this funcion'''
    n_labels = dense_labels.shape[0]
    labels_one_hot = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        labels_one_hot[i,int(dense_labels[i])-1] = 1
    return labels_one_hot

def one_hot_to_dense(one_hot_labels):
    ''' '''
    n_labels = one_hot_labels.shape[0]
    dense_labels = np.argmax(one_hot_labels,axis=1) + 1
    return dense_labels

def split_training_testing(X,Y,percentage_for_testing=10):
    ''' '''
    rand_idx = np.random.permutation(X.shape[0])
    training_up_to_idx = (rand_idx.shape[0]*(100-percentage_for_testing))//100

    X_train = X[rand_idx[:training_up_to_idx],...]
    Y_train = Y[rand_idx[:training_up_to_idx]]
    X_test = X[rand_idx[training_up_to_idx:],...]
    Y_test = Y[rand_idx[training_up_to_idx:]]

    return X_train, Y_train, X_test, Y_test

def _find_idx_start(x_full_length, sequence_length, perc_overlap):
    ''' '''
    i_start = lambda idx : int(idx*(sequence_length-(perc_overlap/100)*sequence_length))
    i_stop = lambda idx : i_start(idx) + sequence_length
        
    i = 0
    idx_start = []
    while (i_stop(i) < x_full_length):
        idx_start.append(i_start(i))
        i = i+1
            
    return idx_start

def generate_sequences(df, well_list, feature_names, sequence_length=51, perc_overlap=70, 
                       with_labels=True, flip_ud=False, scaler=None):
    ''' '''
   
    if scaler is None:
        feature_scaling = [f for f in feature_names if f != 'NM_M']
        scaler = RobustScaler().fit(df.ix[well_list][feature_scaling].values)
    
    X = np.zeros((sequence_length, len(feature_names), 1))
    IDX_abs_pos = np.zeros((1,1))
    if with_labels is True:
        Y = np.zeros((1,1))
        
    
    for well in well_list:
        x = df.ix[well][feature_names].values
        # 5th index is for NM_M feature
        nm_m = np.expand_dims(x[:,5],1)
        x_ = RobustScaler().fit_transform(np.hstack((x[:,:5],np.expand_dims(x[:,-1],1))))
        x = np.hstack((x_[:,:5],nm_m,np.expand_dims(x_[:,-1],1)))
        #x = scaler.transform(x)
        if with_labels is True:
            y = df.ix[well]['Facies'].values
            
        # find sequences that can be fully extracted
        # for the last one take the exact final piece
        N = x.shape[0]   
        idx_start = _find_idx_start(N, sequence_length, perc_overlap)
                    
        # loop over sequences
        for i in idx_start:
            x_sequence = x[i:i+sequence_length,:]
            if with_labels is True:
                y_sequence = y[i:i+sequence_length]
                Y = np.vstack((Y,np.expand_dims(y_sequence,1)))
                
            # generate roll-pad array
            x_sequence_rp, idx_absolute_pos = _generate_roll_pad_array(x_sequence, x, i)        
            X = np.dstack((X,x_sequence_rp))
            IDX_abs_pos = np.vstack((IDX_abs_pos, np.expand_dims(idx_absolute_pos,1)))
            
        # the last sequence
        x_sequence = x[-sequence_length:,:]
        assert x_sequence.shape[0] == sequence_length
        if with_labels is True:
            y_sequence = y[-sequence_length:]
            assert(x_sequence.shape[0] == y_sequence.shape[0])
            Y = np.vstack((Y,np.expand_dims(y_sequence,1)))
            
            
        x_sequence_rp, idx_absolute_pos = _generate_roll_pad_array(x_sequence,x,N-sequence_length)
        X = np.dstack((X,x_sequence_rp))
        IDX_abs_pos = np.vstack((IDX_abs_pos, np.expand_dims(idx_absolute_pos,1)))
        
        
    # from [width=sequence_length, channels=n_facies, batch] to:
    # [batch, heigth=1, width=sequence_length, channels=n_facies]
    X = np.expand_dims(np.swapaxes(np.swapaxes(X[:,:,1:],0,2),1,2),1)
    
    IDX_abs_pos = np.squeeze(IDX_abs_pos[1:]).astype(int)
    if with_labels is True:
        Y = np.squeeze(Y[1:])
        
    if flip_ud is True:
        X = np.concatenate((X,X[:,:,::-1,:]),axis=0)
        #IDX_abs_pos = np.concatenate((IDX_abs_pos,IDX_abs_pos[::-1]),axis=0)
        IDX_abs_pos = np.concatenate((IDX_abs_pos,IDX_abs_pos),axis=0)
        if with_labels is True:
            #Y = np.concatenate((Y,Y[::-1]),axis=0)
            Y = np.concatenate((Y,Y),axis=0)
        
    
                        
    if with_labels is True:
        return X, Y, scaler
    else:
        return X, IDX_abs_pos
    
    #return X, Y if with_labels is True else X, IDX_abs_pos

def _generate_roll_pad_array(x_sequence, x_full, idx_start_seq):
    ''' '''
    n_seq = x_sequence.shape[0]
    assert n_seq % 2 == 1, 'Input sequence must have uneven number of samples'
    seq_mid_idx = n_seq//2
    n_features = x_sequence.shape[1]
    
    X_roll_pad = np.empty((n_seq,n_features,n_seq))
    X_roll_pad[:] = np.NAN
    
    idx_absolute_pos = np.empty((n_seq))
    idx_absolute_pos[:] = np.NAN
    
    
    for i in range(0,n_seq):
        idx_position_in_x_full = idx_start_seq + i
        X_roll_pad[seq_mid_idx,:,i] = x_sequence[i,:]
        idx_absolute_pos[i] = idx_position_in_x_full
        # before
        idx_range_before_in_x_full = range(idx_position_in_x_full-n_seq//2,idx_position_in_x_full)
        for j_local, j_global in enumerate(idx_range_before_in_x_full):
            if j_global < 0:
                X_roll_pad[j_local,:,i] = 0 #before
            else:
                X_roll_pad[j_local,:,i] = x_full[j_global,:]
        # after
        idx_range_after_in_x_full = range(idx_position_in_x_full+1,idx_position_in_x_full+1+n_seq//2)
        for j_local, j_global in enumerate(idx_range_after_in_x_full):
            if j_global >= x_full.shape[0]:
                X_roll_pad[seq_mid_idx+1+j_local,:,i] = 0 #after
            else:
                X_roll_pad[seq_mid_idx+1+j_local,:,i] = x_full[j_global,:]
        
    return X_roll_pad, idx_absolute_pos


def perturbe_label(labels, approximate_neighbors, perturbation_ratio=0.05):
    '''TO DO: vect'''
    N = labels.shape[0]
    labels_perturbed = np.copy(labels) 
    idx_pert = np.random.permutation(np.arange(N))[:int(perturbation_ratio*N)]
    
    for i in idx_pert:
        assert (int(labels[i])-1) >= 0
        labels_perturbed[i] = np.random.choice(approximate_neighbors[int(labels[i]-1)])
    
    return labels_perturbed 