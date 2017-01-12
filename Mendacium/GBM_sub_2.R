#Desktop
setwd('D:/Thanish/D/Thanish Folder/Compeditions/Facies')

set.seed(100)
train_prod = read.csv('facies_vectors.csv')
test_prod = read.csv('nofacies_data.csv')

#Converting the Facies column to factor
train_prod$Facies = as.factor(as.character(train_prod$Facies))

#Removing the rows with NA in PE and fill 0 in relpos
test_prod$X = NULL
train_prod = train_prod[!is.na(train_prod$PE),]

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

#Splitting into train and test
train_local = train_prod[!train_prod$Well.Name %in% c('SHRIMPLIN'),]
test_local  = train_prod[train_prod$Well.Name %in% c('SHRIMPLIN'),]

#====================================================================================================
#H2o model
library(h2o)
start.h2o = h2o.init(nthreads = -1)
train_local_h2o = as.h2o(train_local[!colnames(train_local) %in% c('Well.Name')])
test_local_h2o  = as.h2o(test_local)
test_prod_h2o  = as.h2o(test_prod)

x.indep = colnames(train_local_h2o[,!colnames(train_local_h2o) %in% c('Facies')])
y.dep = 'Facies'

#gbm
set.seed(100)
gbm.local.model.h2o = h2o.gbm(y=y.dep, x=x.indep, training_frame = train_local_h2o,
                              ntree=500)
gbm.local.pred.h2o = h2o.predict(gbm.local.model.h2o, type='class',newdata = test_local_h2o)
gbm.local.pred.h2o = as.data.frame(gbm.local.pred.h2o)

acc_table_gbm_h2o = table(test_local$Facies, gbm.local.pred.h2o$predict)
acc_gbm_h2o = sum(diag(acc_table_gbm_h2o))/nrow(test_local)
acc_gbm_h2o
F1(acc_table_gbm_h2o)


#on prod
gbm.prod.pred.h2o = h2o.predict(gbm.local.model.h2o, type='class',newdata = test_prod_h2o)
gbm.prod.pred.h2o = as.data.frame(gbm.prod.pred.h2o)

#Writing the output file
sub = cbind(test_prod, Facies = gbm.prod.pred.h2o$predict)
write.csv(sub, row.names= F, 'GBM_predicted_facies_2.csv')

