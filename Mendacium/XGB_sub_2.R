setwd('D:/Thanish/D/Thanish Folder/Compeditions/Facies')

#Install if not already present
#install.packages('xgboost')
library(xgboost)

train_prod = read.csv('facies_vectors.csv')
test_prod = read.csv('nofacies_data.csv')

#Converting the columns to factor and numeric
train_prod$Facies = as.factor(as.character(train_prod$Facies))
train_prod$Formation = as.numeric(train_prod$Formation)
test_prod$Formation = as.numeric(test_prod$Formation)

#Removing the rows with NA
test_prod$X = NULL
train_prod = train_prod[!is.na(train_prod$PE),]

#Splitting into train and test
train_row = sample(nrow(train_prod), 0.7*nrow(train_prod), replace=F)
train_local = train_prod[train_row,]
test_local  = train_prod[-train_row,]

#====================================================================================================
#XGB model
train_xgb_local_indep = train_local[,!colnames(train_local) %in% c('Well.Name',
                                                                   'Facies')]
train_xgb_local_label = train_local[,'Facies']
test_xgb_local_indep = test_local[,!colnames(test_local) %in% c('Well.Name',
                                                                'Facies')]
test_xgb_local_label = test_local[,'Facies']
test_xgb_prod_indep = test_prod[,!colnames(test_prod) %in% c('Well.Name')]

xgb.local.model = xgboost(data = as.matrix(train_xgb_local_indep),
                          label = train_xgb_local_label,
                          num_class = 10,
                          nround = 100,
                          max_depth = 6,
                          eta=0.2,
                          objective = 'multi:softmax'
                          )

xgb.local.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_local_indep))
acc_table_xgb = table(test_local$Facies, as.numeric(xgb.local.pred))
acc_table_xgb
acc_xgb = sum(diag(acc_table_xgb))/nrow(test_local)
acc_xgb

xgb.prod.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_prod_indep))

#====================================================================================================

sub = cbind(test_prod, Facies = xgb.prod.pred)
write.csv(sub, row.names= F, 'XGB_predicted_facies_2.csv')
