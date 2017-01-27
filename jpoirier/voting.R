library(dplyr)

# function to calculate the distance between an observation and it's sequence average in data-space
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# obs = a single observation of data
# seq = data averages for a single facies sequence
#
# SUMMARY OF RETURN
# dist = multi-dimensional distance in data-space between given observation and sequence average
# ---------------------------------------------------------------------------------------------------
calcDist <- function(obs, seq) {
    
    # calculate distance between observed data and sequence average
    GR_dist <- obs$GR - seq$GR_mean
    ILD_log10_dist <- obs$ILD_log10_dist[1] - seq$ILD_log10_mean[1]
    dPhi_dist <- obs$DeltaPHI[1] - seq$dPhi_mean[1]
    PHIND_dist <- obs$PHIND[1] - seq$PHI_mean[1]
    PE_dist <- obs$PE[1] - seq$PE_mean[1]
    
    dist <- sum(GR_dist^2, ILD_log10_dist^2, dPhi_dist^2, PHIND_dist^2, PE_dist^2, na.rm=T)^0.5
    dist
}

# function to tally votes weighted using inverse distance weighting (IDW)
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# test = data frame of observations we want to predict facies for
# trainWells = vector of wells used for training
# p = power parameter for the inverse distance weighting
#
# SUMMARY OF RETURN
# votes = data frame/matrix of weighted votes with rows corresponding to test set and columns to 
#         classifications
# ---------------------------------------------------------------------------------------------------
tallyVotes <- function(test, trainWells, p=1.75) {
    votes <- data.frame(matrix(NA, nrow=nrow(test), ncol=9))
    names(votes) <- c('SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS')
    
    for (i in 1:nrow(test)) {
        for (w in trainWells) {
            # retrieve i-th classification c and distance d
            c <- unlist(test[i, paste("Facies", w)])
            d <- unlist(test[i, paste("Dist", w)])
            
            if (!is.na(c) & !is.na(d)) {
                w <- 1 / d^p     # calculate inverse distance weighting
                
                # tally the vote (set as w if it's still NA, otherwise add it to total)
                if (is.na(unlist(votes[i, which(names(votes) %in% c)]))) {
                    votes[i, which(names(votes) %in% c)] <- w
                } else {
                    votes[i, which(names(votes) %in% c)] <- unlist(votes[i, which(names(votes) %in% c)]) + w
                }            
            }
        }
    }
    
    votes
}

# function to elect a classification
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# test = test data frame for which we want to predict facies
# votes = matrix containing rows corresponding to test data and weighted votes for each facies (cols)
#
# SUMMARY OF RETURN
# test = the given test data frame plus the elected/predicted facies
# ---------------------------------------------------------------------------------------------------
electClass <- function(test, votes) {
    test$Predicted <- NA
    
    for (i in 1:nrow(test)) {
        test$Predicted[i] <- names(votes)[which.max(votes[i,])]        
    }
    
    test
}

# function to have every training well to vote on facies for each test observation
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# train = training data frame
# test = test data frame
#
# SUMMARY OF RETURN
# testPrime = test data frame containing votes from training data and their data-space distances to be
#             used in weighting the votes
# ---------------------------------------------------------------------------------------------------
getOutAndVote <- function(train, test) {
    testPrime <- data.frame()
    
    seq <- faciesSequencing(train)
    seq <- mutate(group_by(seq, Formation, Well.Name), FmCumThickness=cumsum(FmRelThickness))
    
    seqWells <- unique(seq$Well.Name)
    testWells <- unique(test$Well.Name)
    
    for (w in testWells) {
        test_w <- test[test$Well.Name == w,]
        
        for (i in 1:nrow(test_w)) {
            for (sw in seqWells) {
                seq_sw <- seq[seq$Well.Name == sw &
                                  seq$Formation == test_w$Formation[i] &
                                  test_w$FmRelDepth[i] <= seq$FmCumThickness,][1,]
                test_w[i, paste("Facies", sw)] <- seq_sw$Facies[1]
                test_w[i, paste("Dist", sw)] <- calcDist(test_w[i,], seq_sw)
            }
        }
        
        testPrime <- rbind(testPrime, test_w)
    }
    
    testPrime
}