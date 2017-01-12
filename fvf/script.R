
library(ggplot2)
library(caret)
library(e1071)
library(randomForest)
library(dplyr)
library(tidyr)
library(rpart)
library(class)
library(gbm)
library(reshape2)


# reading in the training and validation data

input0 <- read.csv("facies_vectors.csv")
valid0 <- read.csv("validation_data_nofacies.csv")

input <- transform(input0, Facies=as.factor(Facies), Formation=as.factor(Formation))
input <- input[complete.cases(input),c(1,2,4:9)]

# creating a test subset from training data

intrain <- createDataPartition(input$Facies, p=0.75, list = FALSE)
train <- input[intrain,]
test <- input[-intrain,]

# fitting and testing a random forest model

fitrf <- randomForest(Facies~., data=train)
tstrf <- predict(fitrf, newdata=test)

print(confusionMatrix(tstrf, test$Facies, mode = "prec_recall"))

# applying the random forest model

valid <- transform(valid0, Formation=as.factor(Formation))
valid <- valid[complete.cases(valid),c(1,3,4:8)]

prdrf <- predict(fitrf, newdata=valid)

# saving the predicted facies

valid0$PredictedFacies <- prdrf
write.csv(valid0, "predicted_facies.csv")

