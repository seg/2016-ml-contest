# function to load data
loadData <- function() {
    fname <- "../facies_vectors.csv"
    data <- read.csv(fname, colClasses=c(rep("factor",3), rep("numeric",6), "factor", "numeric"))
    
    data
}

# function to pre-process the data
cleanData <- function(data) {
    # convert NM_M channel into a binary channel "isMarine"
    data$NM_M <- data$NM_M == "2"
    names(data)[which(names(data) == "NM_M")] <- "isMarine"
    
    if ("Facies" %in% names(data)) {
        # make the Facies channel more descriptive
        levels(data$Facies) <- c("SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS")   
    }
    
    data
}

# function to split the data
splitData <- function(data, testWell) {
    testIndex <- data$Well.Name == testWell
    
    train <- data[!testIndex,]
    test <- data[testIndex,]
    split <- list(train, test)
    
    split
}