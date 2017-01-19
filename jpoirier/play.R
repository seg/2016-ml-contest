play <- function() {
    
    # load data
    data <- read.csv("../facies_vectors.csv")
    
    # preprocess data
    data$Facies <- as.factor(data$Facies)
    levels(data$Facies) <- c("SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS")
    data$NM_M <- data$NM_M == "2"
    names(data)[10] <- "isMarine"
    
    # split data
    inTrain <- data$Well.Name != "SHANKLE"
    train <- data[inTrain,]
    test <- data[!inTrain,]
    
    crossCorrs <- data.frame()
    
    l <- 50     # window length
    n <- 5      # number of horizons to include
    wells <- unique(train$Well.Name)
    
    for (well_i in wells) {
        crossCorrsi <- data.frame()
        train_i <- train[train$Well.Name == well_i,]
        
        # loop through arbitrary horizons in test well
        for (j in 1:floor(nrow(test)/l)) {
            
            top <- ((j-1) * l + 1)
            base <- (j * l)
            test_j <- test[top:base,]
            
            temp <- loopAcrossFeatures(test_j, train_i, 
                                       whichFeatures(train_i, c("GR", "ILD_log10", "DeltaPHI", "PHIND", "PE")), 
                                       crossCorrelate)
            temp$trainWell <- well_i
            temp$testWell <- test$Well.Name[1]
            temp$top <- top
            temp$base <- base
            temp$testCenter <- temp$top + l / 2
            temp$trainCenter <- temp$testCenter + temp$lag
            
            crossCorrsi <- rbind(crossCorrsi, temp)    
        }
        r <- crossCorrsi[order(-crossCorrsi$correlation),]
        r <- r[match(unique(r$top), r$top),]
        r <- r[1:n,]
        
        crossCorrs <- rbind(crossCorrs, r)
    }
    
    crossCorrs

}

whichFeatures <- function(train, testFeatures) {
    
    # ensure features have actual data
    goodFeatures <- names(train)[apply(train, 2, function(x) {sum(is.na(x))==0})]
    
    # certain features we're just not interested in modeling
    badFeatures <- c("Formation", "Well.Name", "Depth")
    if (sum(train$Well.Name == "Recruit F9") == nrow(train)) badFeatures <- c(badFeatures, "RELPOS")
    goodFeatures <- goodFeatures[!goodFeatures %in% badFeatures]
    
    # finally, we only want to include features which also exist in the test data set
    goodFeatures <- goodFeatures[goodFeatures %in% testFeatures]
    
    goodFeatures
}

# function to perform crosscorrelation between two vectors
crossCorrelate <- function(a, b) {
    
    # calculate cross-correlation between vectors a and b
    ccor <- ccf(a, b, lag.max=400, plot=F)
    
    # retrieve the maximum correlation and associated lag
    corr <- max(ccor[["acf"]][,,1])
    lag <- ccor[["lag"]][,,1][which.max(ccor[["acf"]][,,1])]
    
    # return maximum correlation and associated lag
    list(correlation=corr, lag=lag)
}

# apply a function "FUN" over columns of data frames "a" and "b"
# NOTE: FUN must take two arguments, two vectors of data
loopAcrossFeatures <- function(a, b, features, FUN) {
    
    # get list of columns for a and b dataframes
    features_a <- names(a)[names(a) %in% features]
    features_b <- names(b)[names(b) %in% features]
    
    # ensure a and b data frames have the same features
    try ((if (!all.equal(features_a, features_b)) stop("Error! Data frames do not have the same features.")))
    
    # initialize resulting data frame
    r <- data.frame()
    
    # loop through features
    for (feature in features) {
        # retrieve the feature vector of interest from each data frame
        av <- as.data.frame(a[,which(names(a) %in% feature)])
        bv <- as.data.frame(b[,which(names(b) %in% feature)])
        
        # apply the function to the two features, storing in a data frame
        temp <- as.data.frame(FUN(av, bv))
        temp$feature <- feature
        
        # merge result with those of other features
        r <- rbind(temp,r)
    }
    
    r
}