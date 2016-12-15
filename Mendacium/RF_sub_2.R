#Laptop
setwd('E:/Thanish/Data science/Facies')

train_prod = read.csv('facies_vectors.csv')
test_prod = read.csv('nofacies_data.csv')

str(train_prod)
str(test_prod)

#Removing the rows with NA
test_prod$X = NULL
train_prod = train_prod[!is.na(train_prod$PE),]

#Filling up NA with the mean
train_prod$PE[is.na(train_prod$PE)] = mean(train_prod$PE, na.rm = T)

#Converting the Facies column to factor
train_prod$Facies = as.factor(as.character(train_prod$Facies))


#SPlitting into train and test
train_row = sample(nrow(train_prod), 0.7*nrow(train_prod), replace=F)
train_local = train_prod[train_row,]
test_local  = train_prod[-train_row,]
str(train_local)
str(test_local)

#====================================================================================================
#Random Forest 
library(randomForest)
RF.local.model = randomForest(Facies~., data = train_local[!colnames(train_local) %in% c('Well.Name')])
RF.local.pred  = predict(RF.local.model, newdata = test_local)
acc_table = table(RF.local.pred, test_local$Facies)
acc_table
acc = sum(diag(acc_table))/nrow(test_local)
acc

RF.prod.pred  = predict(RF.local.model, newdata = test_prod)

#====================================================================================================

sub = cbind(test_prod, Facies = RF.prod.pred)
write.csv(sub, row.names= F, 'RF_predicted_facies_2.csv')

