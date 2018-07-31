library(plotly)
library(ggplot2)
library(RColorBrewer)
library(plyr)
library(scales)
library(reshape)
library(stargazer)
library(pROC)

line_size <- 0.5
point_size <- 2
base_size <- 26
text_size <- 6
width <- 16
height <- 12

remove_prefix_nr_from_caseid <- function(row) {
  case_id <- row["case_id"]
  parts <- strsplit(case_id, "_")[[1]]
  cut_length <- ifelse(as.numeric(row["nr_events"]) < 2, length(parts), length(parts)-1)
  return(paste(parts[1:cut_length], collapse="_"))
}

setwd("results_detailed")
result_files <- list.files()
data <- data.frame()
for (filename in result_files) {
  tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
  tmp$case_id <- as.character(tmp$case_id)
  tmp$nr_events <- as.numeric(as.character(tmp$nr_events))
  tmp$case_id <- apply(tmp, 1, remove_prefix_nr_from_caseid)
  data <- rbind(data, tmp)
}
setwd("results_lstm_detailed")
result_files <- list.files()
for (filename in result_files) {
  tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
  data <- rbind(data, tmp)
}

data$nr_events <- as.factor(data$nr_events)
data$dataset <- as.character(data$dataset)
data$cls <- as.character(data$cls)
data$method <- as.character(data$method)

data <- subset(data, grepl("calibrated", Method))

data[data$cls=="lstm_calibrated", "cls"] <- "LSTM"
data[data$cls=="rf_calibrated", "cls"] <- "RF"
data[data$cls=="xgboost_calibrated", "cls"] <- "XGB"
data[data$method=="single_agg", "method"] <- "agg"
data[data$method=="prefix_index", "method"] <- "idx_mul"
data[data$method=="single_index", "method"] <- "idx_pad"

data$Method <- paste(data$cls, data$method, sep="_")
data[data$cls=="lstm_calibrated", "Method"] <- "LSTM"

data[data$dataset=="hospital_billing_3", "dataset"] <- "hospital_billing"
data[data$dataset=="traffic_fines_1", "dataset"] <- "traffic_fines"
data[data$dataset=="sepsis_cases_4", "dataset"] <- "sepsis_cases_3"

## Part 1: general comparison
# AUC
dt_aucs <- ddply(data, .(dataset, nr_events, Method), summarize, count=length(actual), auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, predicted))))
write.table(dt_aucs, "aucs_calibrated.csv", sep=";", row.names=FALSE, col.names=TRUE)

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
pdf("images/aucs_calibrated_3cols.pdf", width=width, height=height*4/3)
ggplot(dt_aucs, 
       aes(x=as.numeric(nr_events), y=auc, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Prefix length", y="AUC") + theme(legend.position = 'top',
                                                                                                                                    legend.key.size = unit(1.5, 'lines'))
dev.off()

# stability (expected difference of successive values)
data <- 
  data %>%
  group_by(case_id, dataset, Method) %>%
  mutate(lag.value = dplyr::lag(predicted, n = 1, default = NA))

dt_diffs <- subset(data, !is.na(lag.value))
dt_diffs$diff <- abs(dt_diffs$predicted - dt_diffs$lag.value)

dt_stab <- ddply(dt_diffs, .(dataset, Method), summarize, stability=1-mean(diff))
write.table(dt_stab, "stability_orig.csv", sep=";", row.names=FALSE, col.names=TRUE)

dt_sd_cases <- ddply(data, .(dataset, Method, case_id), summarize, coef_mean_abs=1-mean(abs(diff(predicted))), coef_median_abs=median(abs(diff(predicted))), coef_sd=1-(sd(diff(predicted))), coef_idr=1-(abs(quantile(diff(predicted), 0.9) - quantile(diff(predicted), 0.1))))
dt_sd <- ddply(dt_sd_cases, .(dataset, Method), summarize, mean_mean_abs=mean(coef_mean_abs, na.rm=TRUE))

pdf("images/stability_mean_abs_calibrated_3cols.pdf", width=width, height=height*4/3)
ggplot(dt_sd, aes(x=Method, y=mean_mean_abs, fill=Method)) + geom_bar(stat="identity", color="black") + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free", ncol=3) + geom_text(aes(label=round(mean_mean_abs, 3)), size=5, vjust=-0.25) + 
  scale_fill_manual(values=color_palette) + theme(axis.text.x=element_blank(), legend.position = 'top',
                                                  legend.key.size = unit(1.5, 'lines')) + #ylim(c(0, 1.05)) +
  ylab("Temporal stability") + xlab("") +  coord_cartesian(ylim=c(0.65, 1.05))
dev.off()


# overall AUCs
dt_aucs$auc <- as.numeric(dt_aucs$auc)
dt_aucs$weighted_auc <- dt_aucs$auc * dt_aucs$count
agg_aucs <- ddply(dt_aucs, .(dataset, Method), summarize, avg_weighted_auc=sum(weighted_auc, na.rm=TRUE)/sum(count, na.rm=TRUE), avg_auc=mean(auc, na.rm=TRUE))
head(agg_aucs)

color_palette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
ggplot(agg_aucs, aes(x=Method, y=avg_weighted_auc, fill=Method)) + geom_bar(stat="identity") + 
  theme_bw() + facet_wrap(~dataset, scales="free") + geom_text(aes(label=round(avg_weighted_auc, 3))) + scale_fill_manual(values=color_palette)



## Part 2: inter-run stability (RMSPD)
setwd("results_5runs_auc_detailed")
result_files <- list.files()
dt_runs <- data.frame()
for (filename in result_files) {
  tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
  tmp$method <- "5"
  dt_runs <- rbind(dt_runs, tmp)
}
setwd("results_5runs_auc_rmspd_detailed")
result_files <- list.files()
for (filename in result_files) {
  tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
  tmp$method <- "5_S"
  dt_runs <- rbind(dt_runs, tmp)
}

dt_runs$nr_events <- as.factor(dt_runs$nr_events)
dt_runs$dataset <- as.character(dt_runs$dataset)
data$cls <- as.character(data$cls)
data$method <- as.character(data$method)

data[data$cls=="rf_calibrated", "cls"] <- "RF"
data[data$cls=="xgboost_calibrated", "cls"] <- "XGB"
data$Method <- paste(data$cls, data$method, sep="_")
data <- subset(data, grepl("calibrated", Method))

dt_runs[dt_runs$dataset=="hospital_billing_3", "dataset"] <- "hospital_billing"
dt_runs[dt_runs$dataset=="traffic_fines_1", "dataset"] <- "traffic_fines"
dt_runs[dt_runs$dataset=="sepsis_cases_4", "dataset"] <- "sepsis_cases_3"

dt_runs <- rbind(dt_runs, subset(data.frame(data), !grepl("LSTM", Method) & !grepl("idx", Method))[,-9])
dt_runs[dt_runs$Method=="RF_agg", "Method"] <- "RF"
dt_runs[dt_runs$Method=="XGB_agg", "Method"] <- "XGB"


dt_aucs <- ddply(dt_runs, .(dataset, nr_events, Method), summarize, count=length(actual), auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, predicted))))
write.table(dt_aucs, "aucs_runs.csv", sep=";", row.names=FALSE, col.names=TRUE)
head(dt_aucs)

# overall AUCs
dt_aucs$auc <- as.numeric(dt_aucs$auc)
dt_aucs$weighted_auc <- dt_aucs$auc * dt_aucs$count
agg_aucs <- ddply(dt_aucs, .(dataset, Method), summarize, avg_weighted_auc=sum(weighted_auc, na.rm=TRUE)/sum(count, na.rm=TRUE), avg_auc=mean(auc, na.rm=TRUE)) # weigh by number of samples for each nr_events!!!!
head(agg_aucs)

library(stargazer)

stargazer(cast(agg_aucs, dataset~Method, mean, value="avg_weighted_auc"), summary=FALSE, rownames=FALSE)

dt_sd_cases <- ddply(dt_runs, .(dataset, Method, case_id), summarize, coef_mean_abs=1-mean(abs(diff(predicted))), coef_sd=1-(sd(diff(predicted))), coef_idr=1-(abs(quantile(diff(predicted), 0.9) - quantile(diff(predicted), 0.1))))
dt_sd <- ddply(dt_sd_cases, .(dataset, Method), summarize, mean_mean_abs=mean(coef_mean_abs, na.rm=TRUE))

stargazer(cast(dt_sd, dataset~Method, mean, value="mean_mean_abs"), summary=FALSE, rownames=FALSE)


## Part 3: exponential smoothing
data$nr_events <- as.numeric(data$nr_events)
betas = c(0, 0.1, 0.25, 0.5, 0.75, 0.9)
datasets <- unique(data$dataset)
mets <- unique(data$Method)

for (beta in betas) {
  print(beta)
  for (ds in datasets) {
    print(ds)
    for (met in mets) {
      dt_selected <- data.frame(subset(data, dataset==ds & Method==met)[,names(data)!="lag.value"])
      if (nrow(dt_selected) > 0) {
        smoothed_preds = subset(dt_selected, nr_events == 1)
        smoothed_preds$smoothed_pred <- smoothed_preds$predicted
        for (i in (min(dt_selected$nr_events)+1):max(dt_selected$nr_events)) {
          tmp <- subset(dt_selected, nr_events == i)
          tmp <- merge(tmp, smoothed_preds[smoothed_preds$nr_events==i-1,c("case_id", "smoothed_pred", "Method")], by=c("case_id", "Method"))
          tmp$smoothed_pred = beta*tmp$smoothed_pred + (1-beta)*tmp$predicted
          smoothed_preds <- rbind(smoothed_preds, tmp)
        }
        dt_aucs <- ddply(smoothed_preds, .(dataset, nr_events, Method, cls), summarize, count=length(actual), auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, smoothed_pred))))
        write.table(dt_aucs, sprintf("smoothed_files/aucs_%s_%s_%s.csv", ds, met, beta), sep=";", row.names=FALSE, col.names=TRUE)
        
        dt_sd_cases <- ddply(smoothed_preds, .(dataset, Method, case_id), summarize, coef_mean_abs=1-mean(abs(diff(smoothed_pred))), coef_sd=1-(sd(diff(smoothed_pred))), coef_idr=1-(abs(quantile(diff(smoothed_pred), 0.9) - quantile(diff(smoothed_pred), 0.1))))
        dt_sd <- ddply(dt_sd_cases, .(dataset, Method), summarize, mean_mean_abs=mean(coef_mean_abs, na.rm=TRUE))
        write.table(dt_sd, sprintf("smoothed_files/stab_%s_%s_%s.csv", ds, met, beta), sep=";", row.names=FALSE, col.names=TRUE)
        
      }
    }
  }
}

setwd("smoothed_files")
# AUCS 
smoothed_aucs <- data.frame()
for (beta in betas) {
  result_files <- list.files(".", paste("aucs.*", as.character(beta), "\\.csv", sep=""))
  result_files <- result_files[!grepl("index", result_files)]
  for (filename in result_files) {
    tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
    tmp$beta <- beta
    smoothed_aucs <- rbind(smoothed_aucs, tmp)
  }
}
smoothed_aucs$Method <- as.character(smoothed_aucs$Method)

smoothed_aucs$auc <- as.numeric(smoothed_aucs$auc)
smoothed_aucs$weighted_auc <- smoothed_aucs$auc * smoothed_aucs$count
agg_aucs <- ddply(smoothed_aucs, .(dataset, Method, beta), summarize, avg_auc=sum(weighted_auc, na.rm=TRUE)/sum(count, na.rm=TRUE)) # weigh by number of samples for each nr_events!!!!
head(agg_aucs)

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
pdf("images/aucs_all_3cols.pdf", width=width, height=height*4/3)
ggplot(subset(agg_aucs), aes(x=beta, y=avg_auc, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Alpha", y="AUC") + theme(legend.position = 'top',
                                                                                                                    legend.key.size = unit(1.5, 'lines'))

dev.off()

# stab
smoothed_stab <- data.frame()
for (beta in betas) {
  result_files <- list.files(".", paste("stab_.*", as.character(beta), "\\.csv", sep=""))
  for (filename in result_files) {
    tmp <- read.table(filename, sep=";", header=T, na.strings=c("None"))
    tmp$beta <- beta
    smoothed_stab <- rbind(smoothed_stab, tmp)
  }
}

pdf("images/stability_all_3cols.pdf", width=width, height=height*4/3)
ggplot(smoothed_stab, aes(x=beta, y=mean_mean_abs, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Alpha", y="Temporal stability") + theme(legend.position = 'top',
                                                                                                                                   legend.key.size = unit(1.5, 'lines'))

dev.off()

# stability vs. auc
dt_merged <- merge(agg_aucs, smoothed_stab, by=c("dataset", "Method", "beta"))

pdf("images/stability_vs_auc_3cols.pdf", width=width, height=height*4/3)
ggplot(subset(dt_merged), aes(x=mean_mean_abs, y=avg_auc, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_color_manual(values=color_palette) + labs(x="Temporal stability", y="AUC") + theme(legend.position = 'top',
                                                                                                                                 legend.key.size = unit(1.5, 'lines')) +
  scale_x_continuous(breaks = trans_breaks(identity, identity, n = 3))

dev.off()


## Supplementary

# examples for paper
#ts1 <- seq(0.6, 0.3, by=-0.03)
ts2 <- rep(0.25, 11)
ts3 <- c(rep(0.5, 5), rep(0.9, 6))

set.seed(22)
ts4 <- runif(11, min=0.4, max=1)
ts5 <- sin(1:11)
ts5 <- (0.7-0.1)/(max(ts5)-min(ts5))*(ts5-min(ts5))+0.1
#ts6 <- c(rep(0.25, 4), 0.8, rep(0.25, 4), 0.6, 0.3)

dt_examples <- data.frame(nr_events=1:11, prediction_score=ts2, case="Case A")
dt_examples <- rbind(dt_examples, data.frame(nr_events=1:11, prediction_score=ts3, case="Case B"))
dt_examples <- rbind(dt_examples, data.frame(nr_events=1:11, prediction_score=ts4, case="Case C"))
dt_examples <- rbind(dt_examples, data.frame(nr_events=1:11, prediction_score=ts5, case="Case D"))
dt_examples$type <- "original"

beta = 0.8
rolling_preds = subset(dt_examples, nr_events == 1)
rolling_preds$prediction_score_smoothed <- rolling_preds$prediction_score
for (i in (min(dt_examples$nr_events)+1):max(dt_examples$nr_events)) {
  tmp <- subset(dt_examples, nr_events == i)
  tmp <- merge(tmp, rolling_preds[rolling_preds$nr_events==i-1,c("case", "prediction_score_smoothed")], by="case")
  tmp$prediction_score_smoothed = beta*tmp$prediction_score_smoothed + (1-beta)*tmp$prediction_score
  rolling_preds <- rbind(rolling_preds, tmp)
}
rolling_preds$prediction_score <- rolling_preds$prediction_score_smoothed
rolling_preds$type <- "smoothed"
rolling_preds <- rolling_preds[,-ncol(rolling_preds)]
dt_examples <- rbind(dt_examples, rolling_preds)

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7)]
pdf("images/example_prediction_scores.pdf", width=13, height=6)
ggplot(dt_examples, aes(x=nr_events, y=prediction_score, color=case, shape=case)) + geom_point(size=5) + geom_line(size=1) + 
  theme_bw(base_size=base_size) + theme(legend.position = 'right', legend.key.size = unit(1.5, 'lines')) + 
  guides(color=guide_legend(title=""), shape=guide_legend(title="")) + xlab("Prefix length") + ylab("Prediction score") + 
  scale_color_manual(values=color_palette) + facet_grid(~type) + theme(legend.position="top")
dev.off()



# case lengths
dt_case_lengths <- read.table("case_lengths_with_classes.csv", sep=";", header=T)
head(dt_case_lengths)

dt_case_lengths$label <- as.character(dt_case_lengths$label)
dt_case_lengths[dt_case_lengths$label=="pos", "label"] <- "positive"
dt_case_lengths[dt_case_lengths$label=="neg", "label"] <- "negative"

dt_case_lengths$dataset <- as.character(dt_case_lengths$dataset)
dt_case_lengths[dt_case_lengths$dataset=="hospital_billing_3", "dataset"] <- "hospital_billing"
dt_case_lengths[dt_case_lengths$dataset=="traffic_fines_1", "dataset"] <- "traffic_fines"
dt_case_lengths[dt_case_lengths$dataset=="sepsis_cases_4", "dataset"] <- "sepsis_cases_3"
dt_case_lengths$label[dt_case_lengths$dataset=="sepsis_cases_3" & dt_case_lengths$label=="positive"] <- "neg"
dt_case_lengths$label[dt_case_lengths$dataset=="sepsis_cases_3" & dt_case_lengths$label=="negative"] <- "positive"
dt_case_lengths$label[dt_case_lengths$dataset=="sepsis_cases_3" & dt_case_lengths$label=="neg"] <- "negative"

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7)]
pdf("images/case_length_hist_3cols.pdf", width=width, height=height*4/3)
ggplot(dt_case_lengths, 
       aes(x=case_count, fill=label, group=label)) + geom_histogram(binwidth=1) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=3) + scale_fill_manual(values=c("#0072B2", "#CC79A7")) + labs(x="Case length", y="Count") + theme(legend.position = 'top',
                                                                                                                            legend.key.size = unit(1.5, 'lines'))
dev.off()

# prefix lengths used for training
dt_pref_lengths <- read.table("prefix_lengths_with_classes.csv", sep=";", header=T)

dt_pref_lengths$dataset <- as.character(dt_pref_lengths$dataset)
dt_pref_lengths[dt_pref_lengths$dataset=="hospital_billing_3", "dataset"] <- "hospital_billing"
dt_pref_lengths[dt_pref_lengths$dataset=="traffic_fines_1", "dataset"] <- "traffic_fines"
dt_pref_lengths[dt_pref_lengths$dataset=="sepsis_cases_4", "dataset"] <- "sepsis_cases_3"

head(dt_pref_lengths)

head(dt_aucs)

dt_prefix_lengths_merged <- merge(subset(dt_pref_lengths, data_type=="train" & label == "all"), dt_aucs, by=c("dataset", "nr_events"))

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
pdf("images/n_train_prefixes_auc.pdf", width=width, height=9)
ggplot(dt_prefix_lengths_merged, 
       aes(x=as.numeric(case_count), y=auc, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=20) + xlab("# train prefixes") + ylab("AUC") + 
  scale_color_manual(values=color_palette) + theme(legend.position="top") +
  facet_wrap(~dataset, scales="free")
dev.off()

png("images/prefix_counts_in_train.png", width=1300, height=900)
ggplot(subset(dt_pref_lengths, data_type=="train" & label!="all"), 
       aes(x=as.numeric(nr_events), y=case_count, color=label, shape=label)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + facet_wrap(~dataset, scales="free") + scale_color_manual(values=color_palette[c(6,3)]) + 
  labs(x="Prefix length", y="# prefixes in training set") + theme(legend.position = 'top', legend.key.size = unit(1.5, 'lines'))
dev.off()


# only max length cases
dt_max_prefixes <- ddply(data, .(case_id, dataset), summarize, case_length=max(as.numeric(nr_events)))
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("bpic2012", dataset) | case_length==40)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("bpic2017", dataset) | case_length==20)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("sepsis_cases_1", dataset) | case_length==29)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("sepsis_cases_2", dataset) | case_length==13)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("sepsis_cases_3", dataset) | case_length==22)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("hospital_billing", dataset) | case_length==8)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("traffic_fines", dataset) | case_length==10)
dt_max_prefixes <- subset(dt_max_prefixes, !grepl("production", dataset) | case_length==16)

dt_max_prefixes_merged <- merge(data, dt_max_prefixes, on=c("case_id", "dataset"))

dt_aucs_max_pref <- ddply(dt_max_prefixes_merged, .(dataset, nr_events, Method), summarize, count=length(actual), auc=ifelse(length(unique(actual)) < 2, NA, auc(roc(actual, predicted))))

color_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")[c(1,8,4,6,2,7,3)]
pdf("images/aucs_calibrated_max_prefix.pdf", width=width, height=5)
ggplot(subset(dt_aucs_max_pref, dataset %in% c("bpic2012_cancelled", "sepsis_cases_2", "sepsis_cases_3", "traffic_fines")), 
       aes(x=as.numeric(nr_events), y=auc, color=Method, shape=Method)) + geom_point(size=point_size) + geom_line(size=line_size) + 
  theme_bw(base_size=base_size) + 
  facet_wrap(~dataset, scales="free", ncol=4) + scale_color_manual(values=color_palette) + labs(x="Prefix length", y="AUC") + theme(legend.position = 'top',
                                                                                                                                    legend.key.size = unit(1.5, 'lines'))
dev.off()

# smoothed predictions
beta = 0.8
rolling_preds = subset(dt_examples, nr_events == 1)
rolling_preds$prediction_score_smoothed <- rolling_preds$prediction_score
for (i in (min(dt_examples$nr_events)+1):max(dt_examples$nr_events)) {
  tmp <- subset(dt_examples, nr_events == i)
  tmp <- merge(tmp, rolling_preds[rolling_preds$nr_events==i-1,c("case", "prediction_score_smoothed")], by="case")
  tmp$prediction_score_smoothed = beta*tmp$prediction_score_smoothed + (1-beta)*tmp$prediction_score
  rolling_preds <- rbind(rolling_preds, tmp)
}

png("images/example_smoothed.png", width=width, height=height)
ggplot(rolling_preds, aes(x=nr_events, y=prediction_score_smoothed, color=case)) + geom_point(size=3) + geom_line(size=1) + 
  theme_bw(base_size=base_size) + theme(legend.position = 'right', legend.key.size = unit(1.5, 'lines')) + guides(color=guide_legend(title="")) + 
  xlab("Prefix length") + ylab("Smoothed prediction score")
dev.off()

