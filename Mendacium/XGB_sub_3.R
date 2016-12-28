#Desktop
setwd('D:/Thanish/D/Thanish Folder/Compeditions/Facies')

set.seed(100)
train_prod = read.csv('facies_vectors.csv')
test_prod = read.csv('nofacies_data.csv')

#Converting the Facies column to factor and Formation to numeric
train_prod$Facies = as.factor(as.character(train_prod$Facies))
train_prod$Formation = as.numeric(train_prod$Formation)
test_prod$Formation = as.numeric(test_prod$Formation)

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
library(xgboost)
train_xgb_indep = train_local[,!colnames(train_local) %in% c('Well.Name',
                                                             'Facies')]
train_xgb_label = train_local[,'Facies']
test_xgb_indep = test_local[,!colnames(test_local) %in% c('Well.Name',
                                                          'Facies')]
test_xgb_label = test_local[,'Facies']
test_xgb_prod_indep = test_prod[,!colnames(test_prod) %in% c('Well.Name',
                                                             'Facies')]

set.seed(100)
xgb.local.model = xgboost(data = as.matrix(train_xgb_indep),
                          label = train_xgb_label,
                          num_class = 10,
                          nround = 100,
                          max_depth = 6,
                          eta=0.3,
                          verbose=F,
                          objective = 'multi:softmax'
)
xgb.local.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_indep))
acc_table_xgb = table(test_local$Facies, as.numeric(xgb.local.pred))
acc_xgb = sum(diag(acc_table_xgb))/nrow(test_local)
acc_xgb
F1(acc_table_xgb)

#ON prod set
xgb.prod.pred = predict(xgb.local.model, newdata = as.matrix(test_xgb_prod_indep))

sub = cbind(test_prod, Facies = xgb.prod.pred)
write.csv(sub, row.names= F, 'XGB_predicted_facies_3.csv')
