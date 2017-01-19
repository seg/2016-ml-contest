# Alan Richardson (Ausar Geophysical)
# 2017/01/09
# Simple first attempt using Ridge regression to predict missing PE values, and SVC for the facies

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, grid_search, linear_model, svm, metrics

# Load + preprocessing

train_data = pd.read_csv('../facies_vectors.csv');
train_data = train_data[train_data['Well Name'] != 'Recruit F9']
train_data = train_data.sample(frac=1).reset_index(drop=True)
Ytrain=train_data['Facies']
Xtrain=train_data.drop('Facies',axis=1)
test_data = pd.read_csv('../validation_data_nofacies.csv')
Xtest = test_data.copy()
le=preprocessing.LabelEncoder()
le.fit(Xtrain['Formation'])
for d in [Xtrain, Xtest]:
    d.drop('Well Name', axis=1, inplace=True)
    d['FormationClass']=le.transform(d['Formation'])
    d.drop('Formation', axis=1, inplace=True)

Xtrain=pd.get_dummies(Xtrain,prefix=['Formation', 'NM_M'],columns=['FormationClass', 'NM_M'])
Xtest=pd.get_dummies(Xtest,prefix=['Formation', 'NM_M'],columns=['FormationClass', 'NM_M'])

# Impute missing PE values and standardise data

scalerNoPE = preprocessing.StandardScaler().fit(Xtrain.drop('PE',axis=1))
Xtrain.loc[:,Xtrain.columns != 'PE']=scalerNoPE.transform(Xtrain.drop('PE',axis=1))
XtrainDropNoPE = Xtrain.dropna(axis=0)
scalerPE = preprocessing.StandardScaler().fit(XtrainDropNoPE['PE'].reshape(-1,1))
XtrainDropNoPE.loc[:,'PE']=scalerPE.transform(XtrainDropNoPE['PE'].reshape(-1,1))
Xtrain.loc[~Xtrain.PE.isnull(),'PE']=XtrainDropNoPE.loc[:,'PE']
YPE=XtrainDropNoPE['PE'];
XPE=XtrainDropNoPE.drop('PE',axis=1)

regRidge = grid_search.GridSearchCV(estimator=linear_model.Ridge(), param_grid=[{'alpha': [0.001,0.01,0.1,1,10,100,1000,10000]}], cv=10)
regRidge.fit(XPE, YPE)
Xtrain.loc[Xtrain.PE.isnull(),'PE'] = regRidge.predict(Xtrain.loc[Xtrain.PE.isnull(),:].drop('PE',axis=1))
print regRidge.best_score_
Xtest.loc[:,Xtrain.columns != 'PE']=scalerNoPE.transform(Xtest.drop('PE',axis=1))
Xtest.loc[:,'PE']=scalerPE.transform(Xtest['PE'].reshape(-1,1))

# Predict facies
clf = grid_search.GridSearchCV(estimator=svm.SVC(), param_grid=[{'C': [0.1,0.3,1,3], 'class_weight': [None, 'balanced']}], scoring='f1_micro')
clf.fit(Xtrain, Ytrain)
test_data['Facies']=clf.predict(Xtest)
test_data.to_csv('ar4_predicted_facies_submission001.csv')
