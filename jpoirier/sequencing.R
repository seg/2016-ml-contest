# function to detect the end of a (formation) data frame
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# df = data frame to check if we are looking at the final row
# row = current row of df under investigation
# f = formation
# w = well
# fmThickness = thickness of formation (not just facies sequence)
# facies_prev = previous rows facies
# top = top of facies sequence
# 
# SUMMARY OF RETURN
# seq_df = empty if not the final row of the data frame, otherwise stats on facies sequence
# ---------------------------------------------------------------------------------------------------
changeFaciesEndDF <- function(df, row, f, w, fmThickness, facies_prev, top) {
    seq_df <- data.frame()
    
    # look for end of data frame
    if (row == nrow(df)) {
        base <- df$Depth[row] + .25
        
        temp <- df[df$Depth >= top & df$Depth < base,]
        seq_df <- rbind(seq_df, buildSeqRow(f, w, fmThickness, facies_prev, top, base, temp))
    }
    
    seq_df 
}

# function to detect an intra-formational change in facies
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# df = data frame to check if we are looking at a change in facies for the current row
# row = current row in df under investigation
# f = formation
# w = well
# fmThickness = formation thickness (not facies sequence thickness)
# facies_prev = previous rows facies
# top = top of facies sequence
# 
# SUMMARY OF RETURN
# list:
#   seq_df = empty if no facies change detected, otherwise stats of facies sequence
#   facies_prev = previous rows facies if no facies change, current facies if facies change detected
#   top = top of current facies sequence (if facies change, then top of following facies sequence)
# ---------------------------------------------------------------------------------------------------
changeFacies <- function(df, row, f, w, fmThickness, facies_prev, top) {
    seq_df <- data.frame()
    
    # look for change in facies
    if (df$Facies[row] != facies_prev) {
        base <- df$DepthLag[row] + ((df$Depth[row] - df$DepthLag[row]) / 2)
        
        temp <- df[df$Depth >= top & df$Depth < base,]
        seq_df <- rbind(seq_df, buildSeqRow(f, w, fmThickness, facies_prev, top, base, temp))
        
        # reset values
        facies_prev <- df$Facies[row]
        top <- df$Depth[row] - ((df$Depth[row] - df$DepthLag[row]) / 2)
    }
    
    list(seq_df=seq_df, facies_prev=facies_prev, top=top)
}

# function to build a data frame of each contiguous facies sequence of a given well & formation
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# df = data frame containing all log observations for formation f and well w
# f = formation
# w = well
# 
# SUMMARY OF RETURN
# seq_df = data frame containing data on each contiguous facies sequence for formation f and well w
# ---------------------------------------------------------------------------------------------------
wellSequencing <- function(df, f, w, top, facies_prev) {
    seq_df <- data.frame()
    fmThickness <- max(df$Depth) - min(df$Depth) + .5
    
    for (i in 1:nrow(df)) {
        checkChangeFacies <- changeFacies(df, i, f, w, fmThickness, facies_prev, top)
        seq_df <- rbind(seq_df, checkChangeFacies[["seq_df"]])
        facies_prev <- checkChangeFacies[["facies_prev"]]
        top <- checkChangeFacies[["top"]]
        
        seq_df <- rbind(seq_df, changeFaciesEndDF(df, i, f, w, fmThickness, facies_prev, top))
    }  
    
    seq_df
}

# function to build a data frame of each contiguous facies sequence of a given formation
# ---------------------------------------------------------------------------------------------------
# SUMMARY OF ARGUMENTS
# df = data frame containing all log observations for formation f
# f = formation
#
# SUMMARY OF RETURN
# seq_df = data frame containing data on each contiguous facies sequence for formation f
# ---------------------------------------------------------------------------------------------------
formationSequencing <- function(df, f) {
    seq_df <- data.frame()
    wells <- unique(df$Well.Name)
    
    for (w in wells) {
        df_w <- df[df$Well.Name == w,]
        df_w$DepthLag <- c(NA, df_w$Depth[1:(nrow(df_w)-1)])
        
        seq_df <- rbind(seq_df, wellSequencing(df_w, f, w, df_w$Depth[1] - .25, df_w$Facies[1]))
    } 
    
    seq_df
}