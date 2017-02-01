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
#Converting the NM to 1 and 0
train_test_prod$NM_M = train_test_prod$NM_M-1

#Fillin up Relpos next with 0
train_test_prod$RELPOS_next[is.na(train_test_prod$RELPOS_next)] = 0

#######
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
#Deep Learning
#H2o model
library(h2o)
start.h2o = h2o.init(nthreads = -1)

train_local_h2o = as.h2o(train_local[!colnames(train_local) %in% c('Well.Name')])
test_local_h2o  = as.h2o(test_local)
test_prod_h2o  = as.h2o(test_prod)

x.indep = colnames(train_local_h2o[,!colnames(train_local_h2o) %in% c('Facies')])
y.dep = 'Facies'

set.seed(100)
DL.local.model.h2o = h2o.deeplearning(y = y.dep, x=x.indep, training_frame = train_local_h2o,
                                      overwrite_with_best_model = T,standardize = T,
                                      hidden = c(200,100,200))

DL.local.pred.h2o = h2o.predict(DL.local.model.h2o, type='class',newdata = test_local_h2o)
DL.local.pred.h2o = as.data.frame(DL.local.pred.h2o)
DL.local.pred.h2o$predict = factor(as.character(DL.local.pred.h2o$predict), levels = levels(test_local$Facies))
acc_table_DL_h2o = table(DL.local.pred.h2o$predict, test_local$Facies)
acc_table_DL_h2o
acc_DL_h2o = sum(diag(acc_table_DL_h2o))/nrow(test_local)
acc_DL_h2o
F1(acc_table_DL_h2o)


#On prod
DL.prod.pred.h2o = h2o.predict(DL.local.model.h2o, type='class',newdata = test_prod_h2o)
DL.prod.pred.h2o = as.data.frame(DL.prod.pred.h2o)

#====================================================================================================

sub = cbind(test_prod, Facies = DL.prod.pred.h2o$predict)
write.csv(sub, row.names= F, 'NN_predicted_facies_4.csv')

