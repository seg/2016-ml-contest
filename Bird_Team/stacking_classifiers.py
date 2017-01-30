#title: Stacking Classifiers
#authors: Leo Dreyfus-Schmidt
#tags: stacking
#created_at: 2016-08-08
#updated_at: 2017-01-13

#In the DrivenData competition, we stacked over 10 classifiers for a 2nd-level stacking. 
#We then clean the code and thought it would be better to do it less hacky and more properly for later use.


from __future__ import division
import pandas as pd, numpy as np, sklearn
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

if sklearn.__version__ >= '0.18.1':
    from sklearn.model_selection import KFold
else:
    from sklearn.cross_validation import KFold

def save(model, name, base_path):
    path = base_path+"/"+name+".pkl"    
    joblib.dump(model,path)
    print name + " is saved in " + path

def load(name, base_path):
    path = base_path+"/"+name+".pkl"
    clf_loaded = joblib.load(path)
    print name + " is loaded"
    
    return clf_loaded

class StackedClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, clfs, skf = "KFold", oob_metrics = metrics.accuracy_score, save_dir = None, level2_learner= LogisticRegression()):
        self.level2_learner = level2_learner
        self.clfs = clfs #list of classifiers to be stacked
        self.skf = skf #for strategy-k-folds- can be any sklearn.cross_validation object or a custom list of couples 
        self.oob_metrics = oob_metrics #for now, only works with predict's metric not predict_proba
        self.save_dir = save_dir
       
    def fit_base_learners(self, X_dev, Y_dev):
        """ Fit the base learners from the training set 
            Output the training set for the second learning layer
        """
        self.n_class = len(set(Y_dev[:3000]))
        self.classes_= range(self.n_class)
        self.dblend_train = None
        self.dico_baselearners = {}
        if self.skf == "KFold" :
            if sklearn.__version__ >= '0.18.1':
                self.skf = list(KFold().split(Y_dev))
            else:
                self.skf = KFold(len(Y_dev), n_folds=3)
            
        for j, clf in enumerate(self.clfs):
            print 'Training classifier [%s]' % (j)     
            dblend_train_j = np.zeros((X_dev.shape[0], self.n_class))     
            for i, (train_index, cv_index) in enumerate(self.skf):
                print 'Fold [%s]' % (i)
                
                X_train = X_dev[train_index]
                Y_train = Y_dev[train_index]
                X_valid = X_dev[cv_index]
                Y_valid = Y_dev[cv_index]
                
                print "Fitting the model !"
                base_learner_cv = clone(clf)
                base_learner_cv.fit(X_train, Y_train) 
                    
                Y_test_predict = base_learner_cv.predict(X_valid)          
                score = self.oob_metrics(Y_valid, Y_test_predict)  
                print "{} on the {}-th fold: {}".format(self.oob_metrics.func_name,i, score)
                           
                dblend_train_j[cv_index] = base_learner_cv.predict_proba(X_valid)
                self.name_base_learner_cv = "base_learner_cv_" + str(j)+ "_fold_" + str(i)
                
                if self.save_dir:
                    save(model = base_learner_cv, name = self.name_base_learner_cv , base_path = self.save_dir)
                else:
                    self.dico_baselearners[self.name_base_learner_cv] = base_learner_cv
                    
            #Vertical concat of the dblend_train_j over j
            if self.dblend_train is None:
                self.dblend_train = dblend_train_j
            else:
                self.dblend_train= np.c_[self.dblend_train,dblend_train_j]

        return self.dblend_train
            
    def predict_base_learners(self,X_test):
        """Predict the class probabilities for X and for each base learners.
        """
        self.dblend_test = None
        for j, clf in enumerate(self.clfs):
            print 'Loading Training classifier [%s] ...' % (j)
            dblend_test_j = np.zeros((X_test.shape[0], self.n_class))     
            for i, (train_index, cv_index) in enumerate(self.skf):
                print '... of Fold [%s]' % (i)
                
                if self.save_dir:
                    base_learner_cv_loaded = load(name = self.name_base_learner_cv, base_path = self.save_dir)
                else:
                    base_learner_cv_loaded = self.dico_baselearners[self.name_base_learner_cv]
                dblend_test_j +=  base_learner_cv_loaded.predict_proba(X_test) #for average later
           
            #For each base_learners, averaging the proba of each class per folds
            dblend_test_j = dblend_test_j/len(self.skf)
            if self.dblend_test is None:
                self.dblend_test = dblend_test_j
            else:
                self.dblend_test= np.c_[self.dblend_test,dblend_test_j]
                
        return self.dblend_test

    def fit_level2(self,Y_dev):
        #Fitting level-2 learner
        self.level2_learner.fit(self.dblend_train, Y_dev)
        #save the level2 model in the folder
        if self.save_dir:
            save(model = self.level2_learner, name = "level2_learner" , base_path = self.save_dir)
        else:
            self.dico_baselearners["level2_learner"] = self.level2_learner
        return self

    def predict_level2(self, X_test):
        if self.save_dir:
            level2_learner_loaded = load(name = "level2_learner", base_path = self.save_dir)
        else:
            level2_learner_loaded = self.dico_baselearners["level2_learner"]
        # Predict now
        Y_test_predict = level2_learner_loaded.predict_proba(self.dblend_test)
        return Y_test_predict

    def fit(self, X_dev,Y_dev):
        self.fit_base_learners(X_dev,Y_dev)
        self.fit_level2(Y_dev)
        return self

    def predict_proba(self, X_test):
        self.predict_base_learners(X_test)
        return self.predict_level2(X_test)