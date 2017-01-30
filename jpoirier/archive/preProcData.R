firstDiff <- function(data) {
    
    wells <- unique(data$Well.Name)
    dataPrime <- data.frame()
    
    # loop through wells
    for (well_i in wells) {
        data_i <- data[data$Well.Name == well_i,]
        
        # loop through observations in well_i
        for (j in 1:nrow(data_i)) {
            
            # backward diff and central diff equal zero
            if (j == 1) {
                
                # forward diffs
                data_i[j, "GR_forDiff"] <- (data_i[j+1, "GR"] - data_i[j, "GR"]) 
                data_i[j, "ILD_forDiff"] <- (data_i[j+1, "ILD_log10"] - data_i[j, "ILD_log10"])
                data_i[j, "DeltaPHI_forDiff"] <- (data_i[j+1, "DeltaPHI"] - data_i[j, "DeltaPHI"])
                data_i[j, "PHIND_forDiff"] <- (data_i[j+1, "PHIND"] - data_i[j, "PHIND"]) 
                
                # backward diffs
                data_i[j, "GR_bacDiff"] <- 0
                data_i[j, "ILD_bacDiff"] <- 0
                data_i[j, "DeltaPHI_bacDiff"] <- 0
                data_i[j, "PHIND_bacDiff"] <- 0
                
                # central diffs
                data_i[j, "GR_cenDiff"] <- 0
                data_i[j, "ILD_cenDiff"] <- 0
                data_i[j, "DeltaPHI_cenDiff"] <- 0
                data_i[j, "PHIND_cenDiff"] <- 0
                
                # PE log diff's if PE log is present
                if (c("PE") %in% names(data_i)) {
                    data_i[j, "PE_forDiff"] <- (data_i[j+1, "PE"] - data_i[j, "PE"])
                    data_i[j, "PE_bacDiff"] <- 0
                    data_i[j, "PE_cenDiff"] <- 0
                }
                
            
            # forward diff and central diff equal to zero
            } else if (j == nrow(data_i)) {
                
                # forward diffs
                data_i[j, "GR_forDiff"] <- 0
                data_i[j, "ILD_forDiff"] <- 0
                data_i[j, "DeltaPHI_forDiff"] <- 0
                data_i[j, "PHIND_forDiff"] <- 0
                
                # backward diffs
                data_i[j, "GR_bacDiff"] <- (data_i[j, "GR"] - data_i[j-1, "GR"])
                data_i[j, "ILD_bacDiff"] <- (data_i[j, "ILD_log10"] - data_i[j-1, "ILD_log10"])
                data_i[j, "DeltaPHI_bacDiff"] <- (data_i[j, "DeltaPHI"] - data_i[j-1, "DeltaPHI"])
                data_i[j, "PHIND_bacDiff"] <- (data_i[j, "PHIND"] - data_i[j-1, "PHIND"])
                
                # central diffs
                data_i[j, "GR_cenDiff"] <- 0
                data_i[j, "ILD_cenDiff"] <- 0
                data_i[j, "DeltaPHI_cenDiff"] <- 0
                data_i[j, "PHIND_cenDiff"] <- 0

                # PE log diff's if PE log is present
                if (c("PE") %in% names(data_i)) {
                    data_i[j, "PE_forDiff"] <- 0   
                    data_i[j, "PE_bacDiff"] <- (data_i[j, "PE"] - data_i[j-1, "PE"])
                    data_i[j, "PE_cenDiff"] <- 0
                }
                
            # all may be calulated nicely
            } else {
                
                # forward diffs
                data_i[j, "GR_forDiff"] <- (data_i[j+1, "GR"] - data_i[j, "GR"])
                data_i[j, "ILD_forDiff"] <- (data_i[j+1, "ILD_log10"] - data_i[j, "ILD_log10"])
                data_i[j, "DeltaPHI_forDiff"] <- (data_i[j+1, "DeltaPHI"] - data_i[j, "DeltaPHI"])
                data_i[j, "PHIND_forDiff"] <- (data_i[j+1, "PHIND"] - data_i[j, "PHIND"]) 
                
                # backward diffs
                data_i[j, "GR_bacDiff"] <- (data_i[j, "GR"] - data_i[j-1, "GR"])
                data_i[j, "ILD_bacDiff"] <- (data_i[j, "ILD_log10"] - data_i[j-1, "ILD_log10"])
                data_i[j, "DeltaPHI_bacDiff"] <- (data_i[j, "DeltaPHI"] - data_i[j-1, "DeltaPHI"])
                data_i[j, "PHIND_bacDiff"] <- (data_i[j, "PHIND"] - data_i[j-1, "PHIND"])
                
                # central diffs
                data_i[j, "GR_cenDiff"] <- (data_i[j+1, "GR"] - data_i[j-1, "GR"]) / 2
                data_i[j, "ILD_cenDiff"] <- (data_i[j+1, "ILD_log10"] - data_i[j-1, "ILD_log10"]) / 2
                data_i[j, "DeltaPHI_cenDiff"] <- (data_i[j+1, "DeltaPHI"] - data_i[j-1, "DeltaPHI"]) / 2
                data_i[j, "PHIND_cenDiff"] <- (data_i[j+1, "PHIND"] - data_i[j-1, "PHIND"]) / 2
                
                # PE log diff's if PE log is present
                if (c("PE") %in% names(data_i)) {
                    data_i[j, "PE_forDiff"] <- (data_i[j+1, "PE"] - data_i[j, "PE"]) 
                    data_i[j, "PE_bacDiff"] <- (data_i[j, "PE"] - data_i[j-1, "PE"])
                    data_i[j, "PE_cenDiff"] <- (data_i[j+1, "PE"] - data_i[j-1, "PE"]) / 2
                }
                
            }
            
        }
        
        dataPrime <- rbind(dataPrime, data_i)
    }
    
    dataPrime
}

lagData <- function(data, l) {
    
    wells <- unique(data$Well.Name)
    dataPrime <- data.frame()
    
    for (well_i in wells) {
        data_i = data[data$Well.Name == well_i,]
        
        for (i in (-l/2):(l/2)) {
            if (i < 0) {
                suffix <- paste0("n", abs(i))
            } else {
                suffix <- i
            }
            data_i[,paste0("GR_", suffix)] <- lagpad(data_i$GR, i)
            data_i[,paste0("ILD_log10_", suffix)] <- lagpad(data_i$ILD_log10, i)
            data_i[,paste0("DeltaPHI_", suffix)] <- lagpad(data_i$DeltaPHI, i)
            data_i[,paste0("PHIND_", suffix)] <- lag(data_i$PHIND, i)
            data_i[,paste0("isMarine_", suffix)] <- lag(data_i$isMarine, i)
            
            if (c("PE") %in% names(data_i)) {
                data_i[,paste0("PE_", suffix)] <- lag(data_i$PE, i)
            }
        }
        
        dataPrime <- rbind(dataPrime, data_i)
    }
    
    # do not return original channels, as they are included as lag "0"
    if (c("PE") %in% names(dataPrime)) {
        dataPrime <- subset(dataPrime, select=-c(GR, ILD_log10, DeltaPHI, PHIND, PE, isMarine))
    } else {
        dataPrime <- subset(dataPrime, select=-c(GR, ILD_log10, DeltaPHI, PHIND, isMarine))
    }
    
    dataPrime
}

lagpad <- function(x, k) {
    
    if (!is.vector(x)) 
        stop('x must be a vector')
    if (!is.numeric(x)) 
        stop('x must be numeric')
    if (!is.numeric(k))
        stop('k must be numeric')
    if (1 != length(k))
        stop('k must be a single number')
    
    if (k < 0) {
        tail <- tail(x, (abs(k)+1))
        xPrime <- c(x, rev(tail)[2:length(tail)])
        xPrime <- xPrime[(abs(k)+1):length(xPrime)]
    } else if (k > 0) {
        head <- head(x, (k+1))
        xPrime <- c(rev(head)[1:(length(head)-1)], x)[1:length(x)]
    } else {
        xPrime <- x
    }

    xPrime
}