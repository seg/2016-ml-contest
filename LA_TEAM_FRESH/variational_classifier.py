'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import KFold , StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import numpy as np
np.random.seed(42)
from scipy.spatial import cKDTree

from itertools import groupby

def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 14))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.NM_M, logs.Depth, '-')
    ax[2].plot(logs.ILD_log10, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.Facies  , logs.Depth, '-', color='r')
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
    ax[1].set_xlim(logs.NM_M.min(), logs.NM_M.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.Facies.min(),logs.Facies.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    plt.show()

def occurrence(seq):
    "max/count"
    seq = list(seq)
    max_element = max(seq, key=seq.count)
    return (max_element, seq.count(max_element))

def get_prediction_from_latent_space(x_train_encoded, x_test_encoded, y_train, neighbors=2, leafsize=30):
    latent_space_search_tree = cKDTree(x_train_encoded, leafsize=leafsize)
    y_predicted = []

    for data_point in x_test_encoded:
        distance, id = latent_space_search_tree.query(data_point, k=neighbors, distance_upper_bound=1.0)
        id = id-1
        inv_distance_weighted_metric = {}
        for facies in list(set(y_train[id])):
            inv_distance_weighted_metric[facies] = 0.
            
        
        for dist, point_id in zip(distance, id):
            point = y_train[point_id]
            inv_dist = 1./dist
            inv_distance_weighted_metric[point] += inv_dist
            
        predicted_facies = max(inv_distance_weighted_metric.keys(), key=(lambda key: inv_distance_weighted_metric[key]))
        
        y_predicted.append(predicted_facies)
    return y_predicted
    
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
    
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

#Data Input 
facies_data = pd.read_csv("2016-ml-contest/training_data.csv")
test_data = pd.read_csv("2016-ml-contest/validation_data_nofacies.csv")

test_data_x = test_data[["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M"]]

X = facies_data[["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M"]].bfill()
y = np_utils.to_categorical(facies_data["Facies"].values.astype(np.int32))
concated_data = pd.concat([X, test_data_x])

#Data Scaling
scaler = StandardScaler().fit(concated_data)
X_all_scaled = scaler.transform(concated_data)
X_scaled = X_all_scaled[0:X.shape[0]]
X_test_data_scaled = X_all_scaled[X.shape[0]:]

#Parametrisation
batch_size = 1
nb_classes = y.shape[1]
original_dim = X.shape[1]
latent_dim = 2
intermediate_dim = 128
nb_epoch = 1
epsilon_std = 1.0
neighbors = 3
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='elu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

x_train = X_all_scaled
x_train_test = X_scaled
x_test = X_test_data_scaled

vae.fit(x_train, x_train,
        shuffle=False,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))
        
train_pred = vae.predict(x_train, batch_size=batch_size)

c = Dense(nb_classes, activation="softmax")(z_mean)
classifier = Model(x, c)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(x_train_test, y, shuffle=True, nb_epoch=nb_epoch*5, 
               batch_size=batch_size, validation_data=(x_train_test, y))
y_pred = classifier.predict(x_train_test, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)
y_comp = np.argmax(y, axis=1)
f1_train = f1_score(y_pred, y_comp, average='weighted')
print(confusion_matrix(y_pred, y_comp))
print(f1_train)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
x_train_encoded = encoder.predict(x_train_test, batch_size=batch_size)

plt.figure(figsize=(6, 6))
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_comp)
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=[10]*len(x_data_points[:, 0]), marker="*", s=3)
plt.colorbar()
plt.show() 

y_data_pred = np.argmax(classifier.predict(x_test, batch_size=batch_size), axis=1)

test_data["Facies"] = y_data_pred

# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
test_data.loc[:,'FaciesLabels'] = test_data.apply(lambda row: label_facies(row, facies_labels), axis=1)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors)
    
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors)
    
test_data.to_csv("autoencoder_classifier_prediction.csv")   

"""
#Training
logo = LeaveOneGroupOut()
epochs=1
scores = []
for train, test in logo.split(X_scaled, y, groups=facies_data["Well Name"][facies_data["Facies"]> 3]):
    train_index, test_index = train, test

    x_train, x_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)


    vae.fit(x_train, x_train,
            shuffle=False,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # display a 2D plot of the digit classes in the latent space
    x_train_encoded = encoder.predict(x_train, batch_size=batch_size)
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    x_data_points = encoder.predict(X_test_data_scaled, batch_size=batch_size)

    y_data_train = get_prediction_from_latent_space(x_train_encoded, x_train_encoded, y_train, neighbors=neighbors, leafsize=30)
    f1_train = f1_score(y_train, y_data_train, average='weighted')
    print(confusion_matrix(y_train, y_data_train))
    print(f1_train)
    y_data_test = get_prediction_from_latent_space(x_train_encoded, x_test_encoded, y_train, neighbors=neighbors, leafsize=30)

    f1_test = f1_score(y_test, y_data_test, average='weighted')
    print(confusion_matrix(y_test, y_data_test))
    print(f1_test)
    scores.append(f1_test)
    
print(np.mean(scores), np.min(scores), np.max(scores), np.std(scores))
print(scores)
    

y_data_pred = get_prediction_from_latent_space(x_train_encoded, x_data_points, y_train, neighbors=neighbors, leafsize=30)

plt.figure(figsize=(6, 6))
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train)
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=[10]*len(x_data_points[:, 0]), marker="*", s=3)
plt.colorbar()
plt.show()

test_data["Facies"] = y_data_pred

# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
test_data.loc[:,'FaciesLabels'] = test_data.apply(lambda row: label_facies(row, facies_labels), axis=1)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors)
    
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors)
    
test_data.to_csv("autoencoder_prediction.csv")
"""