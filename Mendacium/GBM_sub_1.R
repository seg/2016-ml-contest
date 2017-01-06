#Predicting the Facies using GBM from H2o package

train_prod = read.csv('facies_vectors.csv')
test_prod = read.csv('nofacies_data.csv')

str(train_prod)
str(test_prod)
summary(train_prod)
summary(test_prod)

#Converting the Facies column to factor
train_prod$Facies = as.factor(as.character(train_prod$Facies))

#Removing the rows with NA
test_prod$X = NULL
train_prod = train_prod[!is.na(train_prod$PE),]

#SPlitting into train and test
train_row = sample(nrow(train_prod), 0.7*nrow(train_prod), replace=F)
train_local = train_prod[train_row,]
test_local  = train_prod[-train_row,]

#Gradient boosting model using H2o library
library(h2o)
start.h2o = h2o.init(nthreads = -1)
train_local_h2o = as.h2o(train_local[!colnames(train_local) %in% c('Well.Name')])
test_local_h2o  = as.h2o(test_local)
test_prod_h2o  = as.h2o(test_prod)

x.indep = colnames(train_local_h2o[,!colnames(train_local_h2o) %in% 'Facies'])
y.dep = 'Facies'

#gbm
gbm.local.model.h2o = h2o.gbm(y=y.dep, x=x.indep, training_frame = train_local_h2o, ntree=500)
gbm.local.pred.h2o = h2o.predict(gbm.local.model.h2o, type='class', newdata = test_local_h2o)
gbm.local.pred.h2o = as.data.frame(gbm.local.pred.h2o)

acc_table_gbm_h2o = table(test_local$Facies, gbm.local.pred.h2o$predict)
acc_table_gbm_h2o
acc_gbm_h2o = sum(diag(acc_table_gbm_h2o))/nrow(test_local)
acc_gbm_h2o

#Predict on blind dataset
gbm.prod.pred.h2o = h2o.predict(gbm.local.model.h2o, type='class',newdata = test_prod_h2o)
gbm.prod.pred.h2o = as.data.frame(gbm.prod.pred.h2o)

#Writing the submission file
sub = cbind(test_prod, Facies = gbm.prod.pred.h2o$predict)
write.csv(sub, row.names= F, 'GBM_predicted_facies_1.csv')



