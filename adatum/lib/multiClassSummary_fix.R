require(compiler)
multiClassSummary <- cmpfun(function (data, lev = NULL, model = NULL){
    
    #Load Libraries
    require(Metrics)
    require(ModelMetrics)
    require(caret)
    
        #Check data
        if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
            stop("levels of observed and predicted data do not match")
        has_class_probs <- all(lev %in% colnames(data))
        if(has_class_probs) {
            ## Overall multinomial loss
            lloss <- mnLogLoss(data = data, lev = lev, model = model)
            requireNamespace("ModelMetrics")
            #Calculate custom one-vs-all ROC curves for each class
            prob_stats <- lapply(levels(data[, "pred"]),
                                 function(x){
                                     #Grab one-vs-all data for the class
                                     obs  <- ifelse(data[,  "obs"] == x, 1, 0)
                                     prob <- data[,x]
                                     AUCs <- try(ModelMetrics::auc(obs, data[,x]), silent = TRUE)
                                     return(AUCs)
                                 })
            roc_stats <- mean(unlist(prob_stats))
        }
        
        #Calculate confusion matrix-based statistics
        CM <- caret::confusionMatrix(data[, "pred"], data[, "obs"])
        
        #Aggregate and average class-wise stats
        #Todo: add weights
        # RES: support two classes here as well
        #browser() # Debug
        if (length(levels(data[, "pred"])) == 2) {
            class_stats <- CM$byClass
        } else {
            class_stats <- colMeans(CM$byClass)
            names(class_stats) <- paste("Mean", names(class_stats))
        }
        
        # Aggregate overall stats
        overall_stats <- if(has_class_probs)
            c(CM$overall, logLoss = as.numeric(lloss), ROC = roc_stats) else CM$overall
        if (length(levels(data[, "pred"])) > 2)
            names(overall_stats)[names(overall_stats) == "ROC"] <- "Mean_AUC"
        
        
        # Combine overall with class-wise stats and remove some stats we don't want
        stats <- c(overall_stats, class_stats)
        stats <- stats[! names(stats) %in% c('AccuracyNull', "AccuracyLower", "AccuracyUpper",
                                             "AccuracyPValue", "McnemarPValue",
                                             'Mean Prevalence', 'Mean Detection Prevalence')]
        
        # Clean names
        names(stats) <- gsub('[[:blank:]]+', '_', names(stats))
        
        # Change name ordering to place most useful first
        # May want to remove some of these eventually
        stat_list <- c("Accuracy", "Kappa", "Mean_F1", "Mean_Sensitivity", "Mean_Specificity",
                       "Mean_Pos_Pred_Value", "Mean_Neg_Pred_Value", "Mean_Detection_Rate",
                       "Mean_Balanced_Accuracy")
        if(has_class_probs) stat_list <- c("logLoss", "Mean_AUC", stat_list)
        if (length(levels(data[, "pred"])) == 2) stat_list <- gsub("^Mean_", "", stat_list)
        
        stats <- stats[c(stat_list)]

        return(stats)
})