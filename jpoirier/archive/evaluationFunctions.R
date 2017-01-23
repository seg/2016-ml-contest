tallyVotes <- function(test, 
                       blendedModel, 
                       classes=c("SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS")
) {
    
    wells <- names(blendedModel[["fits"]])
    
    # initialize data frame for weighted vote tallies with zeros
    votes <- data.frame(matrix(0, nrow = nrow(test), ncol = length(classes)))
    names(votes) <- classes
    
    for (well_i in wells) {
        predictions <- predict(blendedModel[["fits"]][[well_i]], newdata=test)
        w <- blendedModel[["weights"]][[well_i]]
        
        for (i in 1:nrow(test)) {
            # add well weight
            votes[i, which(names(votes) %in% predictions[i])] <- votes[i, which(names(votes) %in% predictions[i])] + w
        }
    }
    
    votes
}

electClass <- function(test, votes) {
    
    for (i in 1:nrow(test)) {
        test$Predicted[i] <- names(votes)[which.max(votes[i,])]
    }
    test$Predicted <- as.factor(test$Predicted)
    levels(test$Predicted) <- c("SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS")
    test$Predicted
}