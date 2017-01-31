# NOTES: 
# -geometric mean doesn't work well for zeros (common with probabilities) AVOID
# -TODO: add weights to the voting in majority_vote() based on model strength

set_seeds <- function(CVfolds, CVreps, tuneLength, init_seed = 42){
    seedNum <- CVfolds * CVreps + 1
    seedLen <- tuneLength^3 #(CVfolds + tuneLength) * 10
    # create manual seeds vector for parallel processing repeatability
    set.seed(init_seed)
    seeds <- vector(mode = "list", length = seedNum)
    for(i in 1:(seedNum-1)) {
        seeds[[i]] <- sample.int(.Machine$integer.max, seedLen)
    }
    ## For the last model:
    seeds[[seedNum]] <- sample.int(.Machine$integer.max, 1)
    return(seeds)
}

geometric_mean = function(x, na.rm=TRUE, zero.propagate = FALSE){
    if(any(x < 0, na.rm = TRUE)){
        return(NaN)
    }
    if(zero.propagate){
        if(any(x == 0, na.rm = TRUE)){
            return(0)
        }
        exp(mean(log(x), na.rm = na.rm))
    } else {
        exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
    }
}

majority_vote <- function(ordered_predList, 
                          reference,
                          voteType = c("count", "prob"), 
                          meanType = c("arithmetic", "geometric"),
                          metric = "F1") {
    voteType <- match.arg(voteType)
    meanType <- match.arg(meanType) # meanType used only for voteType == "prob"
    classes <- levels(reference)
    
    if(voteType == "prob") {
        if(meanType == "geometric") {
            warning(paste("Geometric mean not well-behaved for small or",
                            "zero probabilities. Results may be meaningless."))
            meanFunc <- function(dat) apply(dat, 1, 
                                            function(x) geometric_mean(x))
        } else {
            # meanType == "arithmetic"
            meanFunc <- function (dat) rowSums(dat)/ncol(dat)
        }
        requireNamespace("dplyr")
        probs <- as.data.frame(matrix(nrow = nrow(ordered_predList), ncol = 0))
        
        for(class in classes) {
            probs[[class]] <- ordered_predList %>% 
                dplyr::select(contains(class)) %>% 
                meanFunc()
        }
        #browser()
        votes <- colnames(probs)[max.col(probs, ties.method = "first")]
        #browser()
    } else {
        # voteType == "count"
        votes <- apply(ordered_predList, 1, 
                       function(x) names(which.max(table(x)))) # ties: first
    }
    
    votes <- factor(votes)
    levels(votes) <- classes
    return(votes)
}

averaged_metric <- function(votes, reference, metric = "F1"){
    requireNamespace("caret")
    CM <- caret::confusionMatrix(votes,reference, mode = "everything")
    # metric averaged across all classes
    metric <- colMeans(CM$byClass, na.rm = TRUE)[[metric]]
    return(metric)
}

ordered_predict <- function(modelList, 
                            newdata, 
                            reference, 
                            type = c("raw", "prob"), 
                            metric = "F1") {
    type <- match.arg(type)
    predList <- as.data.frame(lapply(modelList, 
                                     function(x) predict(x, newdata)))
    perfs <- apply(predList, 2, 
                   function(col) {averaged_metric(col, reference, metric)})
    ord <- order(perfs, decreasing = TRUE) # best performer first
    
    if(type == "raw") {
        ordered_predList <- predList[, ord]
    } else { 
        #type == "prob"
        ordered_predList <- as.data.frame(lapply(modelList[ord], 
                                        function(x) predict(x, newdata, type)))
    }
    
    return(ordered_predList)
}

# set seed before calling model_combos() for reproducible results
model_combos <- function(modelList, 
                         reference, 
                         newdata,
                         metric = "F1",
                         voteType = c("count", "prob"),
                         meanType = c("arithmetic", "geometric"),
                         plot = FALSE) {
    requireNamespace("caret")
    voteType <- match.arg(voteType)
    meanType <- match.arg(meanType)
    nModels <- length(modelList)
    
    modelNames <- sapply(modelList, function(x) x$method)
    predList <- ordered_predict(modelList, 
                                newdata, 
                                reference, 
                                metric, 
                                type = ifelse(voteType == "prob","prob","raw")
                                )
    
    comboList <- list()
    for(k in 1:nModels) {
        combos <- combn(modelNames, k, simplify = FALSE)
        comboList <- c(comboList, combos)
    }
    
    scores <- c()
    for(combo in comboList) {
        pat <- paste(combo, collapse = "|")
        
        if((voteType == "count") & (length(combo) < 3)) {
                if(length(combo) == 2) {next} # can't do majority vote with 2
                if(length(combo) == 1) {
                    votes <- predList[, grepl(pat, colnames(predList))]
                }
        } else {
            votes <- predList[, grepl(pat, colnames(predList))] %>% 
                majority_vote(reference, voteType, meanType, metric)
        }
        score <- averaged_metric(votes,reference, metric)
        names(score) <- paste(combo, collapse = ".") 
        scores <- c(scores, score)
        scores <- sort(scores, decreasing = TRUE)
    }

    if(isTRUE(plot)) {
        requireNamespace("ggplot2")
        df <- data.frame(factor(names(scores)), scores)
        names(df) <- c("model", metric)
        p <- ggplot(df) + 
            geom_point(aes(y = reorder(model, get(metric)), x = get(metric))) +
            labs(y = "", x = paste(metric))
        print(p)
    }
    
    return(scores)    
}

plot_wells <- function(dat) {
    p <- ggplot(dat, aes(x = Depth, y = Well.Name)) +
        geom_tile(aes(fill = Facies)) +
        facet_grid(Well.Name ~ ., scales = "free") +
        scale_fill_brewer(name = "Facies", labels = 1:9,palette = "Set1") +
        theme_bw() +
        theme(panel.grid = element_blank(),
              strip.text = element_blank(),
              strip.background = element_blank(),
              plot.title = element_text(hjust = 0.5)) +
        labs(title = "Rock Facies Types and Depths for each Well", 
             y = "", 
             x = "Depth [m]")
    
    print(p)
}

facies_hist <- function(dat) {
    ggplot(dat, aes(x = Facies)) + 
        geom_bar(aes(y = (..count..), fill = Facies)) + 
        scale_fill_brewer(palette = "Set1") + 
        theme_bw() + 
        theme(panel.grid.major.x = element_blank(),
              legend.position = "none",
              plot.title = element_text(hjust = 0.5)) + 
        scale_x_discrete(label = 1:9) + 
        scale_y_continuous(breaks = seq(0, 1000, 100)) + 
        labs(title = "Distribution of Training Data by Facies", 
             x = "Facies Type", 
             y = "Count")
}

