play <- function() {
    library(ggplot2)
    library(cowplot)
    
    data <- read.csv("../facies_vectors.csv")
    data$Facies <- as.factor(data$Facies)
    wells <- unique(data$Well.Name)
    
    levels(data$Facies) <- c("SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS")
    facies_colors <- c('#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D')
    facies_labels <- c('SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS')
    
    for (well_i in wells) {
        data_i <- data[data$Well.Name == well_i,]
        
        if (sum(!is.na(data_i$PE)) == nrow(data_i)) {
            g <- ggplot(data=data_i) + geom_point(aes(x=DeltaPHI, y=ILD_log10, color=Facies, size=PE), alpha=.5) + 
                scale_color_manual(values=facies_colors, drop=F, labels=facies_labels) +
                ylim(0,1)            
        } else {
            g <- ggplot(data=data_i) + geom_point(aes(x=DeltaPHI, y=ILD_log10, color=Facies)) + 
                scale_color_manual(values=facies_colors, drop=F, labels=facies_labels) +
                ylim(0,1)
        }

        save_plot(paste0(well_i, ".png"), g)
    }
}