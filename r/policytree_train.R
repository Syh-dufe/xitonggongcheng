#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "Usage: Rscript r/policytree_train.R <policy-data-dir> <output-dir> <tree-depth>",
    call. = FALSE
  )
}

data_dir <- args[[1]]
if (!dir.exists(data_dir)) {
  stop("policy-data-dir does not exist", call. = FALSE)
}
output_dir <- args[[2]]
tree_depth <- suppressWarnings(as.integer(args[[3]]))
if (is.na(tree_depth) || tree_depth < 1L) {
  stop("tree-depth must be a positive integer", call. = FALSE)
}

optional_library <- Sys.getenv("POLICYTREE_R_LIB", unset = "")
if (nzchar(optional_library)) {
  .libPaths(c(optional_library, .libPaths()))
}
if (!requireNamespace("grf", quietly = TRUE) ||
    !requireNamespace("policytree", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE)) {
  stop(
    paste(
      "Install required packages with:",
      "install.packages(c('grf', 'policytree', 'jsonlite'), repos='https://cloud.r-project.org')"
    ),
    call. = FALSE
  )
}

if (file.exists(file.path(data_dir, "oracle_counterfactuals.csv"))) {
  stop("policy data directory must not contain oracle_counterfactuals.csv", call. = FALSE)
}

train_path <- file.path(data_dir, "policy_training.csv")
test_path <- file.path(data_dir, "policy_test.csv")
manifest_path <- file.path(data_dir, "policy_manifest.json")
if (!all(file.exists(c(train_path, test_path, manifest_path)))) {
  stop("policy data directory must contain training, test, and manifest files", call. = FALSE)
}

manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
if (!identical(manifest$counterfactual_data_used, FALSE)) {
  stop("policy manifest must declare counterfactual_data_used as false", call. = FALSE)
}

train <- utils::read.csv(train_path, check.names = FALSE, stringsAsFactors = FALSE)
test <- utils::read.csv(test_path, check.names = FALSE, stringsAsFactors = FALSE)
required_columns <- c(
  "episode_id", "split", "period", "queue_regular", "queue_specialist",
  "queue_callback_special", "queue_priority", "max_waited_periods",
  "demand_state", "action", "cost"
)
if (!identical(names(train), required_columns) || !identical(names(test), required_columns)) {
  stop("policy input columns do not match the factual export contract", call. = FALSE)
}
if (!identical(sort(unique(train$action)), 0:3) || anyNA(train) || anyNA(test)) {
  stop("training data must contain actions 0, 1, 2, 3 with no missing values", call. = FALSE)
}
if (!all(train$split == "train") || !all(test$split == "test")) {
  stop("training and test files have invalid split labels", call. = FALSE)
}
if (length(intersect(unique(train$episode_id), unique(test$episode_id))) > 0) {
  stop("episode_id occurs in both training and test data", call. = FALSE)
}

feature_columns <- c(
  "period", "queue_regular", "queue_specialist", "queue_callback_special",
  "queue_priority", "max_waited_periods", "demand_state"
)
combined_features <- rbind(train[feature_columns], test[feature_columns])
feature_matrix <- stats::model.matrix(~ . - 1, data = combined_features)
X_train <- feature_matrix[seq_len(nrow(train)), , drop = FALSE]
X_test <- feature_matrix[nrow(train) + seq_len(nrow(test)), , drop = FALSE]
action_factor <- factor(train$action, levels = 0:3)

forest <- grf::multi_arm_causal_forest(
  X = X_train,
  Y = -train$cost,
  W = action_factor,
  seed = 20260924
)
reward_scores <- policytree::double_robust_scores(forest)
tree <- policytree::policy_tree(
  X_train,
  reward_scores,
  depth = tree_depth,
  verbose = FALSE
)
recommended_action <- as.integer(predict(tree, X_test)) - 1L
if (!all(recommended_action %in% 0:3)) {
  stop("policytree returned an action outside 0..3", call. = FALSE)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
recommendations <- data.frame(
  episode_id = test$episode_id,
  period = test$period,
  recommended_action = recommended_action
)
utils::write.csv(
  recommendations,
  file.path(output_dir, "recommendations.csv"),
  row.names = FALSE
)
capture.output(print(tree), file = file.path(output_dir, "policy_tree.txt"))
jsonlite::write_json(
  list(
    method = "grf_multi_arm_causal_forest_plus_policytree",
    tree_depth = tree_depth,
    feature_columns = feature_columns,
    action_encoding = list(original = "0..3", grf_factor_levels = "0..3"),
    train_rows = nrow(train),
    test_rows = nrow(test),
    action_counts_train = as.list(table(factor(train$action, levels = 0:3))),
    observed_action_propensity_quantiles = as.list(
      stats::quantile(forest$W.hat[cbind(seq_len(nrow(train)), as.integer(action_factor))])
    ),
    counterfactual_data_used = FALSE,
    grf_version = as.character(utils::packageVersion("grf")),
    policytree_version = as.character(utils::packageVersion("policytree"))
  ),
  file.path(output_dir, "training_diagnostics.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)
file.copy(manifest_path, file.path(output_dir, "input_policy_manifest.json"), overwrite = TRUE)
