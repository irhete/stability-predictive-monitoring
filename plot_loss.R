setwd("/home/irene/Repos/lstm-predictive-monitoring/multitask_lstm/results/loss_files/")

result_files <- list.files(".", "sepsis")
data <- data.frame()
for (file in result_files) {
  tmp <- read.table(file, sep=";", header=T)
  
  # extract dataset name
  parts <- strsplit(file, "\\.")[[1]]
  parts2 <- strsplit(parts[length(parts)-1], "_")[[1]]
  tmp$dataset <- paste(parts2[-1], collapse="_")
  
  data <- rbind(data, tmp)
} 
head(data)

library(ggplot2)
library(reshape)

dt_melt <- melt(data, id=c("epoch", "params", "dataset"))

ggplot(subset(dt_melt, variable=="train_loss"), aes(x=epoch, y=value, color=params)) + geom_point() + geom_line() + theme_bw() + facet_grid(.~dataset)
ggplot(subset(dt_melt, variable=="val_loss"), aes(x=epoch, y=value, color=params)) + geom_point() + geom_line() + theme_bw() + facet_grid(.~dataset)

ggplot(dt_melt, aes(x=epoch, y=value, color=variable)) + geom_point() + geom_line() + theme_bw() + facet_grid(.~dataset)




