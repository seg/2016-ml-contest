#Office
setwd('D:/Thanish/D/Thanish Folder/Compeditions/Facies')

#Laptop
setwd('E:/Thanish/Data science/Facies')

#Desktop
setwd('D:/Thanish/D/Thanish Folder/Compeditions/Facies')

train_prod = read.csv('facies_vectors.csv')
test_prod = read.csv('nofacies_data.csv')

#Converting the Facies column to factor
train_prod$Facies = as.factor(as.character(train_prod$Facies))

#Adding Facies to test_prod for merging
test_prod$Facies = NA

#Merging the train and test prod
train_test_prod = rbind(train_prod, test_prod)

#Feature engineering
train_test_prod$order = seq(1: nrow(train_test_prod))
head(train_test_prod,20)

#######
#Relpos next
train_test_prod_relpos = train_test_prod[,c('order', 'Well.Name','RELPOS')]
train_test_prod_relpos$order = train_test_prod_relpos$order + 1 
names(train_test_prod_relpos) = c("order", 'Well.Name',"RELPOS_next")
train_test_prod = merge(train_test_prod, train_test_prod_relpos,
                        by.x = c('order','Well.Name'), 
                        by.y = c('order','Well.Name'), 
                        all.x = T)
train_test_prod$RELPOS_next = train_test_prod$RELPOS_next - train_test_prod$RELPOS

#Relpos previous
train_test_prod_relpos$order = train_test_prod_relpos$order -2 
names(train_test_prod_relpos) = c("order", 'Well.Name',"RELPOS_previous")
train_test_prod = merge(train_test_prod, train_test_prod_relpos,
                        by.x = c('order','Well.Name'), 
                        by.y = c('order','Well.Name'), 
                        all.x = T)
train_test_prod$RELPOS_previous = train_test_prod$RELPOS - train_test_prod$RELPOS_previous

#######
#ILD_log10 next
train_test_prod_log10 = train_test_prod[,c('order', 'Well.Name','ILD_log10')]
train_test_prod_log10$order = train_test_prod_log10$order + 1 
names(train_test_prod_log10) = c("order", 'Well.Name',"ILD_log10_next")
train_test_prod = merge(train_test_prod, train_test_prod_log10,
                        by.x = c('order','Well.Name'), 
                        by.y = c('order','Well.Name'), 
                        all.x = T)
train_test_prod$ILD_log10_next = train_test_prod$ILD_log10_next - train_test_prod$ILD_log10

#ILD_log10 previous
train_test_prod_log10$order = train_test_prod_log10$order -2 
names(train_test_prod_log10) = c("order", 'Well.Name',"ILD_log10_previous")
train_test_prod = merge(train_test_prod, train_test_prod_log10,
                        by.x = c('order','Well.Name'), 
                        by.y = c('order','Well.Name'), 
                        all.x = T)
train_test_prod$ILD_log10_previous = train_test_prod$ILD_log10 - train_test_prod$ILD_log10_previous

train_test_prod$order = NULL


######################################################################
#Multiclass F1
F1 = function(M)
{
  precision = NULL
  recall = NULL
  for (i in 1:min(dim(M)))
  {
    precision[i] = M[i,i]/sum(M[,i])
    recall[i] = M[i,i]/sum(M[i,])
  }
  F1 = 2*(precision*recall)/(precision+recall)
  F1[is.na(F1)] = 0
  return(sum(F1)/max(dim(M)))
}
######################################################################

#Converting the NM to factor
train_test_prod$NM_M = as.factor(as.character(train_test_prod$NM_M-1))

#Fillin up Relpos next and previous with 0
train_test_prod$RELPOS_next[is.na(train_test_prod$RELPOS_next)] = 0
train_test_prod$RELPOS_previous[is.na(train_test_prod$RELPOS_previous)] = 0
train_test_prod$ILD_log10_next[is.na(train_test_prod$ILD_log10_next)] = 0
train_test_prod$ILD_log10_previous[is.na(train_test_prod$ILD_log10_previous)] = 0

#######
#Removing the rows with NA
library(missForest)
temp = missForest(train_test_prod[,!colnames(train_test_prod) %in% c('Facies')], 
                  verbose = TRUE, 
                  variablewise = TRUE, 
                  #ntree=20,
                  maxiter = 5
)
train_test_prod = cbind(temp$ximp,train_test_prod[c('Facies')])
######
#Splitting up the train an test prod
train_prod = train_test_prod[!is.na(train_test_prod$Facies),]
test_prod  = train_test_prod[is.na(train_test_prod$Facies),]
test_prod$Facies = NULL
str(train_prod)
str(test_prod)

#Splitting into train and test local

train_local = train_prod[!train_prod$Well.Name %in% c('SHRIMPLIN'),]
test_local  = train_prod[train_prod$Well.Name %in% c('SHRIMPLIN'),]

str(train_local)
str(test_local)
summary(train_local)
summary(train_local)


#====================================================================================================
#Creating SVM
library(e1071)
set.seed(100)
SVM.local.model = svm(Facies~., data = train_local[!colnames(train_local) %in% c(#'Formation',
  #'Depth',
  'Well.Name'
)])
SVM.local.pred = predict(SVM.local.model, newdata = test_local)
acc_table_SVM = table(SVM.local.pred, test_local$Facies)
acc_table_SVM
acc_SVM = sum(diag(acc_table_SVM))/nrow(test_local)
acc_SVM
F1(acc_table_SVM)
macroF1(acc_table_SVM)

#On Prod dataset
SVM.prod.pred = predict(SVM.local.model, newdata = test_prod)
acc_table_SVM = table(SVM.prod.pred, test_prod$Facies)
acc_table_SVM
acc_SVM = sum(diag(acc_table_SVM))/nrow(test_prod)
acc_SVM

#====================================================================================================
#Rpart
library(rpart)
rpart.local.model = rpart(Facies~., data = train_local[!colnames(train_local) %in% c(#'Formation',
  #'Depth',
  'Well.Name'
)])
rpart.local.pred = predict(rpart.local.model, type='class',newdata = test_local)
acc_table_rpart = table(rpart.local.pred, test_local$Facies)
acc_table_rpart
acc_rpart = sum(diag(acc_table_rpart))/nrow(test_local)
acc_rpart

#On Prod dataset
rpart.prod.pred = predict(rpart.local.model, newdata = test_prod)

#====================================================================================================
#Random Forest 
library(randomForest)
set.seed(100)
RF.local.model = randomForest(Facies~.,data = train_local[,!colnames(train_local) %in% c(
  #'Formation',
  #'Depth',
  #'RELPOS_next',
  'RELPOS_previous',
  'ILD_log10_next',
  'ILD_log10_previous',
  'Well.Name'
)])
RF.local.pred  = predict(RF.local.model, newdata = test_local)
acc_table_RF = table(RF.local.pred, test_local$Facies)
#acc_table_RF
acc_RF = sum(diag(acc_table_RF))/nrow(test_local)
acc_RF
F1(acc_table_RF)

RF.prod.pred  = predict(RF.local.model, newdata = test_prod)

#====================================================================================================
#H2o model
library(h2o)
start.h2o = h2o.init(nthreads = -1)
train_local_h2o = as.h2o(train_local[!colnames(train_local) %in% c('Well.Name')])
test_local_h2o  = as.h2o(test_local)
test_prod_h2o  = as.h2o(test_prod)

x.indep = colnames(train_local_h2o[,!colnames(train_local_h2o) %in% c(#'RELPOS_next',
  'RELPOS_previous',
  'ILD_log10_next',
  'ILD_log10_previous',
  'Facies')])
y.dep = 'Facies'

#gbm
gbm.local.model.h2o = h2o.gbm(y=y.dep, x=x.indep, training_frame = train_local_h2o,
                              ntree=500)
gbm.local.pred.h2o = h2o.predict(gbm.local.model.h2o, type='class',newdata = test_local_h2o)
gbm.local.pred.h2o = as.data.frame(gbm.local.pred.h2o)

acc_table_gbm_h2o = table(test_local$Facies, gbm.local.pred.h2o$predict)
#acc_table_gbm_h2o
acc_gbm_h2o = sum(diag(acc_table_gbm_h2o))/nrow(test_local)
acc_gbm_h2o
F1(acc_table_gbm_h2o)
macroF1(acc_table_gbm_h2o)

gbm.prod.pred.h2o = h2o.predict(gbm.local.model.h2o, type='class',newdata = test_prod_h2o)
gbm.prod.pred.h2o = as.data.frame(gbm.prod.pred.h2o)

#====================================================================================================
library(xgboost)
library(dummies)

train_local_dum = dummy.data.frame(train_local[!colnames(train_local) %in% c('Well.Name','Facies')], sep='_')
train_local = cbind(as.data.frame(train_local_dum), Facies = train_local[,'Facies'])
test_local_dum = dummy.data.frame(test_local[!colnames(test_local) %in% c('Well.Name','Facies')], sep="_")
test_local = cbind(as.data.frame(test_local_dum), Facies = test_local[,'Facies'])
str(train_local)
test_prod_dum = dummy.data.frame(test_prod[!colnames(test_prod) %in% c('Well.Name','Facies')], sep="_")
test_prod = data.frame(test_prod_dum)
str(test_prod)

train_xgb_indep = train_local[,!colnames(train_local) %in% c(#'Formation', 
                                                              'Well.Name',
                                                              #'RELPOS_next',
                                                              'RELPOS_previous',
                                                              'ILD_log10_next',
                                                              'ILD_log10_previous',
                                                              'Phi_diff',
                                                              'Facies')]
train_xgb_label = train_local[,'Facies']
test_xgb_indep = test_local[,!colnames(test_local) %in% c(#'Formation', 
                                                            'Well.Name',
                                                            #'RELPOS_next',
                                                            'RELPOS_previous',
                                                            'ILD_log10_next',
                                                            'ILD_log10_previous',
                                                            'Phi_diff',
                                                            'Facies')]
test_xgb_label = test_local[,'Facies']
test_prod_xgb_indep = test_prod[,!colnames(test_prod) %in% c(#'Formation', 
                                                              'Well.Name',
                                                              #'RELPOS_next',
                                                              'RELPOS_previous',
                                                              'ILD_log10_next',
                                                              'ILD_log10_previous',
                                                              'Phi_diff',
                                                              'Facies')]

#xgb.train
#Eval metric
evalacc <- function(preds, real) 
{ 
  labels <- getinfo(real, "label")
  labels <- factor(labels, levels = levels(train_local$Facies))
  acc_table = table(preds, labels)
  Acc = F1(acc_table)
  return(list(metric = "F1", value = Acc))
}

dtrain_local = xgb.DMatrix(data = as.matrix(train_xgb_indep), label = as.matrix(train_xgb_label))
dtest_local  = xgb.DMatrix(data = as.matrix(test_xgb_indep) , label = as.matrix(test_xgb_label))

watchlist <- list(test=dtest_local, train=dtrain_local)
set.seed(100)
xgb.local.model = xgb.train(data = dtrain_local,
                            watchlist=watchlist,
                            num_class = 10,
                            nround = 100,
                            max_depth = 6,
                            eta=0.3,
                            objective = 'multi:softmax',
                            #colsample_bytree = 0.8,
                            #subsample = 0.8,
                            #early.stop.round = 10,
                            #maximize = T,
                            eval_metric= evalacc
)
xgb.local.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_indep))
acc_table_xgb = table(test_local$Facies, as.numeric(xgb.local.pred))
acc_table_xgb
acc_xgb = sum(diag(acc_table_xgb))/nrow(test_local)
acc_xgb
F1(acc_table_xgb)

xgb.importance(model = xgb.local.model, feature_names = colnames(train_xgb_indep))
unique(train_local$Facies)

xgb.prod.pred = predict(xgb.local.model, newdata = as.matrix(test_prod_xgb_indep))
#====================================================================================================

ensemble_DF = data.frame(XGB = (xgb.local.pred),
                         RF   = as.numeric(RF.local.pred),
                         SVM  = as.numeric(SVM.local.pred)
                         #rpart   = as.numeric(rpart.local.pred)
                         #gbm_h2o = as.numeric(gbm.local.pred.h2o$predict)
                         
)

#Prod
ensemble_DF = data.frame(XGB = (xgb.prod.pred),
                         RF   = as.numeric(RF.prod.pred),
                         SVM  = as.numeric(SVM.prod.pred)
                         )

head(ensemble_DF,20)

#Finding the mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

predcited = NULL
for (i in 1:nrow(ensemble_DF))
{
  v = unlist(ensemble_DF[i,])
  predcited[i] = getmode(v)  
}

ensemble_DF$predicted = predcited
acc_table_ensemble = table(ensemble_DF$predicted, (test_local$Facies))
acc_table_ensemble
acc_ensemble = sum(diag(acc_table_ensemble))/nrow(test_local)
acc_ensemble
F1(acc_table_ensemble)

#====================================================================================================

sub = cbind(test_prod, Facies = ensemble_DF$predicted)
write.csv(sub, row.names= F, 'Ensemble_1.csv')

