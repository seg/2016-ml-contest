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

#Splitting up the train an test prod
train_prod = train_test_prod[!is.na(train_test_prod$Facies),]
test_prod  = train_test_prod[is.na(train_test_prod$Facies),]
test_prod$Facies = NULL

#Splitting into train and test local
train_local = train_prod[!train_prod$Well.Name %in% c('SHRIMPLIN'),]
test_local  = train_prod[train_prod$Well.Name %in% c('SHRIMPLIN'),]

#====================================================================================================
library(xgboost)

train_local$Formation = as.numeric(train_local$Formation)
test_local$Formation = as.numeric(test_local$Formation)
train_local$NM_M = as.numeric(as.character(train_local$NM_M))
test_local$NM_M = as.numeric(as.character(test_local$NM_M))
test_prod$Formation = as.numeric(test_prod$Formation)
test_prod$NM_M = as.numeric(as.character(test_prod$NM_M))


train_xgb_indep = train_local[,!colnames(train_local) %in% c(#'Formation', 
                                                              #'RELPOS_next',
                                                              'Well.Name',
                                                              'Facies')]
train_xgb_label = train_local[,'Facies']
test_xgb_indep = test_local[,!colnames(test_local) %in% c(#'Formation', 
                                                          'Well.Name',
                                                          #'RELPOS_next',
                                                          'Facies')]
test_xgb_label = test_local[,'Facies']
test_xgb_prod_indep = test_prod[,!colnames(test_prod) %in% c(#'Formation', 
                                                                'Well.Name',
                                                                #'RELPOS_next',
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


#Cross validation
dtrain_local = xgb.DMatrix(data = as.matrix(train_xgb_indep), label = as.matrix(train_xgb_label))
dtest_local  = xgb.DMatrix(data = as.matrix(test_xgb_indep) , label = as.matrix(test_xgb_label))
watchlist <- list(test=dtest_local, train=dtrain_local)
xgb.local.model = xgb.train(data = dtrain_local,
                            watchlist=watchlist,
                            num_class = 10,
                            nround = 75,
                            max_depth = 6,
                            eta=0.3,
                            objective = 'multi:softmax',
                            #colsample_bytree = 0.8,
                            #subsample = 0.8,
                            early.stop.round = 10,
                            maximize = T,
                            eval_metric= evalacc
)
xgb.local.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_indep))
acc_table_xgb = table(test_local$Facies, as.numeric(xgb.local.pred))
acc_table_xgb
acc_xgb = sum(diag(acc_table_xgb))/nrow(test_local)
acc_xgb
F1(acc_table_xgb)

xgb.local.model = xgboost(data = as.matrix(train_xgb_indep),
                          label = train_xgb_label,
                          num_class = 10,
                          nround = 19,
                          max_depth = 6,
                          eta=0.3,
                          #colsample_bytree = 0.8,
                          #subsample = 0.8,
                          verbose=F,
                          objective = 'multi:softmax'
)
xgb.local.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_indep))
acc_table_xgb = table(test_local$Facies, as.numeric(xgb.local.pred))
acc_table_xgb
acc_xgb = sum(diag(acc_table_xgb))/nrow(test_local)
acc_xgb
F1(acc_table_xgb)


xgb.prod.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_prod_indep))

#====================================================================================================

sub = cbind(test_prod, Facies = xgb.prod.pred)
write.csv(sub, row.names= F, 'XGB_predicted_facies_8.csv')
