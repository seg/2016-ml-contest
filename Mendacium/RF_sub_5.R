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
#Random Forest 
library(randomForest)
set.seed(100)
RF.local.model = randomForest(Facies~., data = train_local[!colnames(train_local) %in% c(
                                                            #'Formation',
                                                            #'Depth',
                                                            'Well.Name'
                                                            )])
RF.local.pred  = predict(RF.local.model, newdata = test_local)
acc_table_RF = table(RF.local.pred, test_local$Facies)
acc_RF = sum(diag(acc_table_RF))/nrow(test_local)
acc_RF
F1(acc_table_RF)

#On prod set
RF.prod.pred  = predict(RF.local.model, newdata = test_prod)

#Writing the output file
sub = cbind(test_prod, Facies = RF.prod.pred)
write.csv(sub, row.names= F, 'RF_predicted_facies_5.csv')

