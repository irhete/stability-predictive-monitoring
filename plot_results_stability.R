library(plotly)
library(ggplot2)
library(RColorBrewer)
library(plyr)
library(scales)
library(reshape)
library(stargazer)


# stability
setwd("/home/irene/Repos/predictive-monitoring-benchmark/results_detailed/")
result_files <- list.files(".", "bpic2012|bpic2017|hospital_billing_3|traffic_fines_1|production|sepsis")
result_files <- result_files[!grepl("complete", result_files)]
result_files <- result_files[!grepl("sample", result_files)]

remove_prefix_nr_from_caseid <- function(row) {
  case_id <- row["case_id"]
  parts <- strsplit(case_id, "_")[[1]]
  cut_length <- ifelse(as.numeric(row["nr_events"]) < 2, length(parts), length(parts)-1)
  return(paste(parts[1:cut_length], collapse="_"))
}

data <- data.frame()
for (filename in result_files) {
  tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
  if (!grepl("lstm", filename)) {
    tmp$case_id <- as.character(tmp$case_id)
    tmp$nr_events <- as.numeric(as.character(tmp$nr_events))
    tmp$case_id <- apply(tmp, 1, remove_prefix_nr_from_caseid)
  }
  data <- rbind(data, tmp)
}

data <- subset(data, dataset != "sepsis_cases_3")
data <- subset(data, params != "cluster_laststate")
data$nr_events <- as.factor(data$nr_events)
data$params <- as.character(data$params)
data[data$params=="pd_fixed_trainratio80_outcome_all_data_singletask", "params"] <- "lstm"
data[data$params=="pd_fixed_trainratio80_outcome_all_data_singletask_timedistributed", "params"] <- "lstm_seq2seq"
data[data$params=="lstm_final", "params"] <- "lstm"

head(data)
nrow(data)

# ROC
library(pROC)
library(plyr)

dt_aucs <- ddply(data, .(dataset, nr_events, params, cls), summarize, count=length(actual), auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, predicted))))
write.table(dt_aucs, "../stability_analysis/aucs_orig.csv", sep=";", row.names=FALSE, col.names=TRUE)
head(dt_aucs)
nrow(dt_aucs)

ggplot(dt_aucs, aes(x=as.numeric(nr_events), y=auc, color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")


# stability
dt_stability_cases <- ddply(data, .(dataset, params, cls, case_id), summarize, std=sd(diff(predicted)))
dt_stability_cases <- dt_stability_cases[!is.na(dt_stability_cases),]
dt_stability <- ddply(dt_stability_cases, .(dataset, params, cls), summarize, mean_std=mean(std, na.rm=TRUE), std_std=sd(std, na.rm=TRUE))
#dt_stability <- dt_stability[!is.na(dt_stability),]
dt_stability <- dt_stability[1:nrow(dt_stability)-1,]
write.table(dt_stability, "../stability_analysis/instability_orig.csv", sep=";", row.names=FALSE, col.names=TRUE)

ggplot(dt_stability, aes(x=factor(params), y=mean_std, fill=params, group=params)) + geom_bar(stat="identity") + theme_bw() + facet_wrap(~dataset, scales="free")


# exponential smoothing
tmp <- subset(data, grepl("Application_1000386745", case_id) & cls=="rf" & params=="prefix_index" & dataset=="bpic2017_accepted")

data$nr_events <- as.numeric(data$nr_events)
betas = c(0.1, 0.25, 0.5, 0.75, 0.9)
datasets <- unique(data$dataset)
#betas <- c(0.1)
#datasets <- c("production")
for (beta in betas) {
  print(beta)
  for (ds in datasets) {
    print(ds)
    dt_selected <- subset(data, dataset==ds)
    smoothed_preds = subset(dt_selected, nr_events == 1)
    smoothed_preds$smoothed_pred <- smoothed_preds$predicted
    smoothed_pred_aucs <- ddply(smoothed_preds, .(dataset, nr_events, params, cls), summarize, auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, smoothed_pred))))
    for (i in (min(dt_selected$nr_events)+1):max(dt_selected$nr_events)) {
      tmp <- subset(dt_selected, nr_events == i)
      tmp <- merge(tmp, smoothed_preds[smoothed_preds$nr_events==i-1,c("case_id", "smoothed_pred", "params")], by=c("case_id", "params"))
      tmp$smoothed_pred = beta*tmp$smoothed_pred + (1-beta)*tmp$predicted
      
      #tmp$smoothed_pred = tmp$smoothed_pred / (1-beta^i)
      #roc_obj <- roc(tmp$actual, tmp$smoothed_pred)
      #dt_auc <- ddply(tmp, .(dataset, nr_events, params, cls), summarize, auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, smoothed_pred))))
      #dt_auc <- data.frame(auc=auc(roc_obj), nr_events=i, prediction_method=paste("smoothed", beta, sep=""))
      #smoothed_pred_aucs <- rbind(smoothed_pred_aucs, dt_auc)
      smoothed_preds <- rbind(smoothed_preds, tmp)
    }
    dt_aucs <- ddply(smoothed_preds, .(dataset, nr_events, params, cls), summarize, count=length(actual), auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, smoothed_pred))))
    write.table(dt_aucs, sprintf("../stability_analysis/aucs_%s_%s.csv", ds, beta), sep=";", row.names=FALSE, col.names=TRUE)
    
    dt_stability_cases <- ddply(smoothed_preds, .(dataset, params, cls, case_id), summarize, std=sd(diff(smoothed_pred)))
    dt_stability_cases <- dt_stability_cases[!is.na(dt_stability_cases),]
    dt_stability <- ddply(dt_stability_cases, .(dataset, params, cls), summarize, mean_std=mean(std, na.rm=TRUE), std_std=sd(std, na.rm=TRUE))
    dt_stability <- dt_stability[1:nrow(dt_stability)-1,]
    write.table(dt_stability, sprintf("../stability_analysis/instability_%s_%s.csv", ds, beta), sep=";", row.names=FALSE, col.names=TRUE)
  }
}


# plotting

dt_aucs <- read.table("/home/irene/Repos/predictive-monitoring-benchmark/stability_analysis/aucs_orig.csv", sep=";", header=T)
head(dt_aucs)

png("/home/irene/Dropbox/stability-predictive-monitoring/images/aucs_orig.png", width=1300, height=900)
ggplot(dt_aucs, aes(x=as.numeric(nr_events), y=auc, color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")
dev.off()

dt_stability <- read.table("/home/irene/Repos/predictive-monitoring-benchmark/stability_analysis/instability_orig.csv", sep=";", header=T)
head(dt_stability)

png("/home/irene/Dropbox/stability-predictive-monitoring/images/stability_orig.png", width=1300, height=900)
ggplot(dt_stability, aes(x=factor(params), y=1-mean_std, fill=params, group=params)) + geom_bar(stat="identity", color="black") + 
  theme_bw() + facet_wrap(~dataset, scales="free") + scale_y_continuous(limits=c(0.6,1), oob=rescale_none) + geom_text(aes(label=round(1-mean_std, 3)))
dev.off()


# plotly

datasets <- c("bpic2012", "bpic2017", "sepsis", "traffic", "production", "hospital")
for (ds in datasets) {
  p <- ggplot(subset(dt_aucs, grepl(ds, dataset)), aes(x=as.numeric(nr_events), y=auc, color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")
  g <- ggplotly(p)
  link <- plotly_POST(g, filename = paste("stability_aucs", ds, sep="_"))
}


setwd("/home/irene/Repos/predictive-monitoring-benchmark/stability_analysis/")

# AUCS 
betas <- c(0.1, 0.25, 0.5, 0.75, 0.9)
smoothed_aucs <- data.frame()
for (beta in betas) {
  result_files <- list.files(".", paste("aucs.*", as.character(beta), sep=""))
  for (filename in result_files) {
    tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
    tmp$beta <- beta
    smoothed_aucs <- rbind(smoothed_aucs, tmp)
  }
}
tmp <- read.table("/home/irene/Repos/predictive-monitoring-benchmark/stability_analysis/aucs_orig.csv", sep=";", header=T)
tmp$beta <- 0
smoothed_aucs <- rbind(smoothed_aucs, tmp)
smoothed_aucs <- subset(smoothed_aucs, dataset != "production" | nr_events != 17)

head(smoothed_aucs)

smoothed_aucs$auc <- as.numeric(smoothed_aucs$auc)
smoothed_aucs$weighted_auc <- smoothed_aucs$auc * smoothed_aucs$count

agg_aucs <- ddply(smoothed_aucs, .(dataset, params, cls, beta), summarize, avg_auc=sum(weighted_auc)/sum(count)) # weigh by number of samples for each nr_events!!!!
head(agg_aucs)

png("/home/irene/Dropbox/stability-predictive-monitoring/images/aucs_all.png", width=1300, height=900)
ggplot(agg_aucs, aes(x=beta, y=avg_auc, color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")
dev.off()

# Stability
betas <- c(0.1, 0.25, 0.5, 0.75, 0.9)
smoothed_stds <- data.frame()
for (beta in betas) {
  result_files <- list.files(".", paste("instability.*", as.character(beta), sep=""))
  for (filename in result_files) {
    tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
    tmp$beta <- beta
    smoothed_stds <- rbind(smoothed_stds, tmp)
  }
}
tmp <- read.table("/home/irene/Repos/predictive-monitoring-benchmark/stability_analysis/instability_orig.csv", sep=";", header=T)
tmp$beta <- 0
smoothed_stds <- rbind(smoothed_stds, tmp)

head(smoothed_stds)

ggplot(subset(smoothed_stds, beta==0.9), aes(x=factor(params), y=mean_std, fill=params, group=params)) + geom_bar(stat="identity", color="black") + theme_bw() + facet_wrap(~dataset, scales="free")

png("/home/irene/Dropbox/stability-predictive-monitoring/images/stability_all.png", width=1300, height=900)
ggplot(subset(smoothed_stds), aes(x=beta, y=1-mean_std, color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")
dev.off()


ds <- "sepsis_cases_1"
ds <- "bpic2012_declined"
ds <- "sepsis_cases_2"
ggplot(subset(smoothed_aucs, dataset==ds) , aes(x=as.numeric(nr_events), y=as.numeric(auc), color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~beta, scales="free")

png("/home/irene/Dropbox/stability-predictive-monitoring/images/aucs_smoothed.png", width=1300, height=900)
ggplot(subset(smoothed_aucs, beta==0.9) , aes(x=as.numeric(nr_events), y=as.numeric(auc), color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")
dev.off()


# merged
dt_merged <- merge(agg_aucs, smoothed_stds, by=c("dataset", "params", "cls", "beta"))
head(dt_merged)

png("/home/irene/Dropbox/stability-predictive-monitoring/images/stability_vs_auc.png", width=1300, height=900)
ggplot(dt_merged , aes(y=avg_auc, x=1-mean_std, color=params)) + geom_point() + geom_line() + theme_bw() + facet_wrap(~dataset, scales="free")
dev.off()


# max AUC and stability reached with each method
max_aucs <- ddply(agg_aucs, .(dataset, params), summarize, max_auc=max(avg_auc))
max_aucs_casted <- cast(max_aucs, dataset~params)
stargazer(max_aucs_casted, summary=FALSE)

smoothed_stds$stability <- 1 - smoothed_stds$mean_std
max_stabilities <- ddply(smoothed_stds, .(dataset, params), summarize, max_auc=max(stability))
max_stabilities_casted <- cast(max_stabilities, dataset~params)
stargazer(max_stabilities_casted, summary=FALSE)
