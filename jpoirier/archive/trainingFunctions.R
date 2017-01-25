library(caret)

trainSingleModel <- function(data) {
    fitControl <- trainControl(method="repeatedcv", number=10, repeats=10)
    
    if (sum(is.na(data$PE_0)) > 0 & c("PE_n15") %in% names(data)) {
        data <- subset(data, select=-c(PE_n15, PE_n14, PE_n13, PE_n12, PE_n11, PE_n10, PE_n9, PE_n8, PE_n7, PE_n6, PE_n5,
                                       PE_n4, PE_n3, PE_n2, PE_n1, PE_0, PE_1, PE_2, PE_3, PE_4, PE_5, PE_6, PE_7, PE_8,
                                       PE_9, PE_10, PE_11, PE_12, PE_13, PE_14, PE_15))
    }
    data$Facies <- factor(data$Facies)
    
    fit <- train(Facies ~ ., data=subset(data, select=-c(Well.Name, Depth)),
                 method="rf", metric="Kappa", tuneLength=10,
                 trControl=fitControl
    )
    
    fit
}

trainBlendedModel <- function(data) {
    fits <- list()
    
    wells <- unique(data$Well.Name)
    
    for (well_i in wells) {
        if (well_i == "Recruit F9") {
            # use all observations; subset of cols data (recruited data have no spatial context)
            data_i <- subset(data, select=c(Facies, Formation, Well.Name, Depth, 
                                            GR_0, ILD_log10_0, DeltaPHI_0, PHIND_0, isMarine_0, RELPOS))
        } else {
            # use well_i data
            data_i <- data[data$Well.Name == well_i,]
        }
        
        fits[[well_i]] <- trainSingleModel(data_i)
    }
    
    fits
}

weightBlendedModel <- function(train, test_iso, p=.25, recruit_wgt=.5) {
    weights <- list()
    
    train_wells <- unique(train$Well.Name)
    
    for (well_i in train_wells) {
        train_i <- train[train$Well.Name == well_i,]
        train_iso <- max(train_i$Depth) - min(train_i$Depth)
        
        # avoid singularity if wells cover same depths
        if (abs(test_iso - train_iso) <= 1) {
            weights[[well_i]] <- 1
        } else if (well_i == "Recruit F9") {
            weights[[well_i]] <- recruit_wgt
        } else {
            weights[[well_i]] <- 1 / (abs(test_iso - train_iso))^p
        }
    }
    
    weights
}