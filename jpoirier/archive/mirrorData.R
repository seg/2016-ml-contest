# subset data - mirroring edges as needed
subsetData <- function(data, top, base) {
    
    # requested subset within data range
    if (top >= 1 & base <= nrow(data)) {
        subset <- data[top:base,]
        
    # requested data occurs before top - mirror data near top
    } else if (top < 1) {
        s1 <- data[2:(abs(top)+2),]
        s1$Depth <- data$Depth[1] - abs(data$Depth[1] - s1$Depth)
        s1$RELPOS <- data$RELPOS[1] + abs(data$RELPOS[1] - s1$RELPOS)
        s1 <- s1[order(s1$Depth),]
        
        s2 <- data[1:base,]
        
        subset <- rbind(s1, s2)
        
    # requested data occurs after base - mirror data near base
    } else if (base > nrow(data)) {
        s1 <- data[top:nrow(data),]
        
        s2 <- data[(nrow(data)-(base-nrow(data))):(nrow(data)-1),]
        s2$Depth <- data$Depth[nrow(data)] + abs(data$Depth[nrow(data)] - s2$Depth)
        s2$RELPOS <- data$RELPOS[nrow(data)] - abs(data$RELPOS[nrow(data)] - s2$RELPOS)
        s2 <- s2[order(s2$Depth),]
        
        subset <- rbind(s1, s2)
    }
    
    subset
}