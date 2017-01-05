#Laptop
setwd('E:/Thanish/Data science/Facies')

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

#Removing the rows with NA in PE
train_test_prod = train_test_prod[!is.na(train_test_prod$PE),]

#######
#Splitting up the train an test prod
train_prod = train_test_prod[!is.na(train_test_prod$Facies),]
test_prod  = train_test_prod[is.na(train_test_prod$Facies),]
test_prod$Facies = NULL

#Splitting into train and test local

train_local = train_prod[!train_prod$Well.Name %in% c('SHRIMPLIN'),]
test_local  = train_prod[train_prod$Well.Name %in% c('SHRIMPLIN'),]

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
macroF1(acc_table_RF)

#====================================================================================================

sub = cbind(test_prod, Facies = RF.prod.pred)
write.csv(sub, row.names= F, 'RF_predicted_facies_1.csv')


