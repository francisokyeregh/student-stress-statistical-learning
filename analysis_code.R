###############################################################
# An Interpretable Statistical Learning Framework for Binary Classification
# An Application to Student Stress Prediction
###############################################################

rm(list = ls())
set.seed(500)

###############################################################
# 1. Install and Load Packages
###############################################################

required_packages <- c(
  "tidyverse", "janitor", "skimr", "caret", "glmnet", "pROC",
  "PRROC", "ranger", "xgboost", "vip", "pdp", "fastshap",
  "naniar", "corrplot", "broom"
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages, dependencies = TRUE)
}

invisible(lapply(required_packages, library, character.only = TRUE))

###############################################################
# 2. Set Project Directory
###############################################################

project_dir <- "C:/Users/kokye/OneDrive/Desktop/FSU_Dept Statistics/Summer 2026/research2"

setwd(project_dir)

output_dir <- project_dir

dir.create(file.path(output_dir, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "models"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "results"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "supplementary"), recursive = TRUE, showWarnings = FALSE)

###############################################################
# 3. Read Dataset
###############################################################

input_file <- file.path(project_dir, "student-lifestyle-and-stress-dataset.csv")

if (!file.exists(input_file)) {
  input_file <- file.path(project_dir, "student-lifestyle-and-stress-dataset(1).csv")
}

if (!file.exists(input_file)) {
  stop("Dataset not found. Please check the file name in the research2 folder.")
}

data_raw <- read.csv(input_file, stringsAsFactors = FALSE)

capture.output(
  str(data_raw),
  file = file.path(output_dir, "tables", "data_structure.txt")
)

###############################################################
# 4. Clean Variable Names and Format Variables
###############################################################

data <- data_raw %>%
  janitor::clean_names()

data <- data %>%
  mutate(
    student_type = as.factor(student_type),
    month = as.factor(month),
    stress_level = factor(
      stress_level,
      levels = c(0, 1),
      labels = c("Low_Stress", "High_Stress")
    )
  )

###############################################################
# 5. Class Distribution
###############################################################

class_distribution <- data %>%
  count(stress_level) %>%
  mutate(percent = round(100 * n / sum(n), 2))

write.csv(
  class_distribution,
  file.path(output_dir, "tables", "Table2_ClassDistribution.csv"),
  row.names = FALSE
)

###############################################################
# 6. Missing Data Summary
###############################################################

missing_summary <- data %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_count"
  ) %>%
  mutate(missing_percent = round(100 * missing_count / nrow(data), 2))

write.csv(
  missing_summary,
  file.path(output_dir, "tables", "MissingDataSummary.csv"),
  row.names = FALSE
)

png(
  file.path(output_dir, "figures", "Figure_MissingDataPattern.png"),
  width = 1200,
  height = 800
)

print(
  naniar::vis_miss(data) +
    ggtitle("Missing Data Pattern")
)

dev.off()

###############################################################
# 7. Impute Missing Values
###############################################################

get_mode <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

data_imputed <- data

numeric_vars <- names(data_imputed)[sapply(data_imputed, is.numeric)]

for (v in numeric_vars) {
  data_imputed[[v]][is.na(data_imputed[[v]])] <-
    median(data_imputed[[v]], na.rm = TRUE)
}

categorical_vars <- names(data_imputed)[sapply(data_imputed, is.factor)]

for (v in categorical_vars) {
  data_imputed[[v]][is.na(data_imputed[[v]])] <- get_mode(data_imputed[[v]])
  data_imputed[[v]] <- droplevels(data_imputed[[v]])
}

write.csv(
  data_imputed,
  file.path(output_dir, "tables", "Clean_Imputed_Dataset.csv"),
  row.names = FALSE
)

###############################################################
# 8. Descriptive Statistics
###############################################################

desc_stats <- skimr::skim(data_imputed)

capture.output(
  desc_stats,
  file = file.path(output_dir, "tables", "Table1_DescriptiveStatistics.txt")
)

numeric_summary_by_stress <- data_imputed %>%
  group_by(stress_level) %>%
  summarise(
    across(
      where(is.numeric),
      list(
        mean = mean,
        sd = sd,
        median = median,
        min = min,
        max = max
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

write.csv(
  numeric_summary_by_stress,
  file.path(output_dir, "tables", "NumericSummary_ByStressLevel.csv"),
  row.names = FALSE
)

###############################################################
# 9. Boxplots by Stress Level
###############################################################

num_vars <- names(data_imputed)[sapply(data_imputed, is.numeric)]

for (v in num_vars) {
  p <- ggplot(data_imputed, aes(x = stress_level, y = .data[[v]])) +
    geom_boxplot() +
    labs(
      title = paste("Distribution of", v, "by Stress Level"),
      x = "Stress Level",
      y = v
    ) +
    theme_minimal()
  
  ggsave(
    file.path(output_dir, "figures", paste0("Boxplot_", v, ".png")),
    p,
    width = 8,
    height = 5,
    dpi = 300
  )
}

###############################################################
# 10. Correlation Matrix and Heatmap
###############################################################

cor_data <- data_imputed %>%
  select(where(is.numeric))

cor_matrix <- cor(cor_data, use = "complete.obs")

write.csv(
  cor_matrix,
  file.path(output_dir, "tables", "Correlation_Matrix.csv")
)

png(
  file.path(output_dir, "figures", "Figure2_CorrelationHeatmap.png"),
  width = 1000,
  height = 800
)

corrplot::corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.cex = 0.8
)

dev.off()

###############################################################
# 11. Train-Test Split
###############################################################

train_index <- caret::createDataPartition(
  data_imputed$stress_level,
  p = 0.70,
  list = FALSE
)

train_data <- data_imputed[train_index, ]
test_data  <- data_imputed[-train_index, ]

train_data$stress_level <- relevel(train_data$stress_level, ref = "High_Stress")
test_data$stress_level  <- relevel(test_data$stress_level, ref = "High_Stress")

write.csv(
  train_data,
  file.path(output_dir, "tables", "Training_Data.csv"),
  row.names = FALSE
)

write.csv(
  test_data,
  file.path(output_dir, "tables", "Testing_Data.csv"),
  row.names = FALSE
)

###############################################################
# 12. Cross-Validation Control
###############################################################

cv_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = FALSE
)

###############################################################
# 13. Model 1: Logistic Regression
###############################################################

model_logit <- train(
  stress_level ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = cv_control
)

###############################################################
# 14. Model 2: Ridge Regression
###############################################################

ridge_grid <- expand.grid(
  alpha = 0,
  lambda = 10^seq(-4, 1, length = 50)
)

model_ridge <- train(
  stress_level ~ .,
  data = train_data,
  method = "glmnet",
  metric = "ROC",
  trControl = cv_control,
  tuneGrid = ridge_grid,
  preProcess = c("center", "scale")
)

###############################################################
# 15. Model 3: LASSO Regression
###############################################################

lasso_grid <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-4, 1, length = 50)
)

model_lasso <- train(
  stress_level ~ .,
  data = train_data,
  method = "glmnet",
  metric = "ROC",
  trControl = cv_control,
  tuneGrid = lasso_grid,
  preProcess = c("center", "scale")
)

###############################################################
# 16. Model 4: Elastic Net Regression
###############################################################

enet_grid <- expand.grid(
  alpha = seq(0.1, 0.9, by = 0.2),
  lambda = 10^seq(-4, 1, length = 30)
)

model_enet <- train(
  stress_level ~ .,
  data = train_data,
  method = "glmnet",
  metric = "ROC",
  trControl = cv_control,
  tuneGrid = enet_grid,
  preProcess = c("center", "scale")
)

###############################################################
# 17. Model 5: Random Forest
###############################################################

rf_grid <- expand.grid(
  mtry = c(2, 3, 4, 5),
  splitrule = "gini",
  min.node.size = c(5, 10, 20)
)

model_rf <- train(
  stress_level ~ .,
  data = train_data,
  method = "ranger",
  metric = "ROC",
  trControl = cv_control,
  tuneGrid = rf_grid,
  importance = "impurity",
  num.trees = 500
)


###############################################################
# 19. Save Models
###############################################################

saveRDS(model_logit, file.path(output_dir, "models", "Model_LogisticRegression.rds"))
saveRDS(model_ridge, file.path(output_dir, "models", "Model_Ridge.rds"))
saveRDS(model_lasso, file.path(output_dir, "models", "Model_LASSO.rds"))
saveRDS(model_enet,  file.path(output_dir, "models", "Model_ElasticNet.rds"))
saveRDS(model_rf,    file.path(output_dir, "models", "Model_RandomForest.rds"))


###############################################################
# 20. Model Evaluation Function
###############################################################

evaluate_model <- function(model, test_data, model_name) {
  
  prob <- predict(model, newdata = test_data, type = "prob")[, "High_Stress"]
  pred <- predict(model, newdata = test_data)
  
  cm <- confusionMatrix(
    pred,
    test_data$stress_level,
    positive = "High_Stress"
  )
  
  roc_obj <- pROC::roc(
    response = test_data$stress_level,
    predictor = prob,
    levels = c("Low_Stress", "High_Stress"),
    direction = "<",
    quiet = TRUE
  )
  
  auc_value <- as.numeric(pROC::auc(roc_obj))
  
  actual_binary <- ifelse(test_data$stress_level == "High_Stress", 1, 0)
  
  brier_score <- mean((prob - actual_binary)^2)
  
  pr <- PRROC::pr.curve(
    scores.class0 = prob[actual_binary == 1],
    scores.class1 = prob[actual_binary == 0],
    curve = FALSE
  )
  
  data.frame(
    Model = model_name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall = as.numeric(cm$byClass["Recall"]),
    F1 = as.numeric(cm$byClass["F1"]),
    ROC_AUC = auc_value,
    PR_AUC = pr$auc.integral,
    Brier_Score = brier_score
  )
}

###############################################################
# 21. Compare Model Performance
###############################################################

performance_table <- bind_rows(
  evaluate_model(model_logit, test_data, "Logistic Regression"),
  evaluate_model(model_ridge, test_data, "Ridge Regression"),
  evaluate_model(model_lasso, test_data, "LASSO Regression"),
  evaluate_model(model_enet,  test_data, "Elastic Net"),
  evaluate_model(model_rf,    test_data, "Random Forest"),
)

write.csv(
  performance_table,
  file.path(output_dir, "tables", "Table3_ModelPerformanceComparison.csv"),
  row.names = FALSE
)

###############################################################
# 22. ROC Curves
###############################################################

roc_data <- list(
  "Logistic Regression" = predict(model_logit, test_data, type = "prob")[, "High_Stress"],
  "Ridge Regression" = predict(model_ridge, test_data, type = "prob")[, "High_Stress"],
  "LASSO Regression" = predict(model_lasso, test_data, type = "prob")[, "High_Stress"],
  "Elastic Net" = predict(model_enet, test_data, type = "prob")[, "High_Stress"],
  "Random Forest" = predict(model_rf, test_data, type = "prob")[, "High_Stress"]
)

roc_list <- lapply(roc_data, function(prob) {
  pROC::roc(
    test_data$stress_level,
    prob,
    levels = c("Low_Stress", "High_Stress"),
    direction = "<",
    quiet = TRUE
  )
})

png(
  file.path(output_dir, "figures", "Figure3_ROC_Curves_AllModels.png"),
  width = 1000,
  height = 800
)

plot(
  roc_list[[1]],
  main = "ROC Curves for Student Stress Prediction"
)

for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], add = TRUE)
}

legend(
  "bottomright",
  legend = paste(
    names(roc_list),
    "AUC =",
    round(sapply(roc_list, pROC::auc), 3)
  ),
  lwd = 2,
  cex = 0.8
)

dev.off()

###############################################################
# 23. Logistic Regression Odds Ratios
###############################################################

final_logit <- glm(
  stress_level ~ .,
  data = train_data,
  family = binomial
)

odds_ratios <- broom::tidy(
  final_logit,
  exponentiate = TRUE,
  conf.int = TRUE
)

write.csv(
  odds_ratios,
  file.path(output_dir, "tables", "LogisticRegression_OddsRatios.csv"),
  row.names = FALSE
)

###############################################################
# 24. Variable Importance: Random Forest
###############################################################

rf_importance <- vip::vi(model_rf)

write.csv(
  rf_importance,
  file.path(output_dir, "tables", "RandomForest_VariableImportance.csv"),
  row.names = FALSE
)

p_rf_vip <- vip::vip(model_rf, num_features = 15) +
  ggtitle("Random Forest Variable Importance")

ggsave(
  file.path(output_dir, "figures", "RandomForest_VariableImportance.png"),
  p_rf_vip,
  width = 8,
  height = 6,
  dpi = 300
)


###############################################################
# 26. Bootstrap Stability Selection Using LASSO
###############################################################

B <- 500

x_train <- model.matrix(stress_level ~ ., data = train_data)[, -1]
y_train <- ifelse(train_data$stress_level == "High_Stress", 1, 0)

selected_matrix <- matrix(
  0,
  nrow = B,
  ncol = ncol(x_train)
)

colnames(selected_matrix) <- colnames(x_train)

for (b in 1:B) {
  
  set.seed(1000 + b)
  
  boot_index <- sample(seq_len(nrow(x_train)), replace = TRUE)
  
  x_boot <- x_train[boot_index, ]
  y_boot <- y_train[boot_index]
  
  cv_fit <- cv.glmnet(
    x = x_boot,
    y = y_boot,
    family = "binomial",
    alpha = 1,
    standardize = TRUE,
    nfolds = 10
  )
  
  coef_b <- coef(cv_fit, s = "lambda.min")
  
  selected_vars <- rownames(coef_b)[which(as.vector(coef_b) != 0)]
  selected_vars <- setdiff(selected_vars, "(Intercept)")
  
  selected_matrix[b, selected_vars] <- 1
}

stability_results <- data.frame(
  Predictor = colnames(selected_matrix),
  Selection_Frequency = colMeans(selected_matrix)
) %>%
  arrange(desc(Selection_Frequency)) %>%
  mutate(
    Stability_Category = case_when(
      Selection_Frequency >= 0.80 ~ "Stable",
      Selection_Frequency >= 0.50 ~ "Moderate",
      TRUE ~ "Weak"
    )
  )

write.csv(
  stability_results,
  file.path(output_dir, "tables", "Table4_LASSO_BootstrapStabilitySelection.csv"),
  row.names = FALSE
)

p_stability <- ggplot(
  stability_results,
  aes(
    x = reorder(Predictor, Selection_Frequency),
    y = Selection_Frequency
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Bootstrap Stability Selection Frequencies",
    x = "Predictor",
    y = "Selection Frequency"
  ) +
  theme_minimal()

ggsave(
  file.path(output_dir, "figures", "Figure4_LASSO_StabilitySelection.png"),
  p_stability,
  width = 9,
  height = 7,
  dpi = 300
)

###############################################################
# 28. SHAP Analysis Using fastshap for Random Forest
###############################################################

prediction_wrapper_rf <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "High_Stress"]
}

set.seed(500)

shap_train <- train_data %>%
  select(-stress_level)

shap_sample <- shap_train %>%
  sample_n(size = min(1000, nrow(shap_train)))

shap_values_rf <- fastshap::explain(
  object = model_rf,
  X = shap_sample,
  pred_wrapper = prediction_wrapper_rf,
  nsim = 50,
  adjust = TRUE
)

shap_importance_rf <- data.frame(
  Feature = colnames(shap_values_rf),
  Mean_Absolute_SHAP = apply(abs(shap_values_rf), 2, mean)
) %>%
  arrange(desc(Mean_Absolute_SHAP))

write.csv(
  shap_importance_rf,
  file.path(output_dir, "tables", "Table5_RandomForest_SHAPImportance.csv"),
  row.names = FALSE
)

p_shap_importance_rf <- ggplot(
  shap_importance_rf,
  aes(
    x = reorder(Feature, Mean_Absolute_SHAP),
    y = Mean_Absolute_SHAP
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Random Forest SHAP-Based Feature Importance",
    x = "Feature",
    y = "Mean Absolute SHAP Value"
  ) +
  theme_minimal()

ggsave(
  file.path(output_dir, "figures", "Figure5_RandomForest_SHAPImportance.png"),
  p_shap_importance_rf,
  width = 8,
  height = 6,
  dpi = 300
)

saveRDS(
  shap_values_rf,
  file.path(output_dir, "models", "RandomForest_fastshap_values.rds")
)

###############################################################
# 29. Model Comparison Figures
###############################################################

p_auc <- ggplot(
  performance_table,
  aes(x = reorder(Model, ROC_AUC), y = ROC_AUC)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Model Comparison by ROC-AUC",
    x = "Model",
    y = "ROC-AUC"
  ) +
  theme_minimal()

ggsave(
  file.path(output_dir, "figures", "ModelComparison_ROC_AUC.png"),
  p_auc,
  width = 8,
  height = 5,
  dpi = 300
)

p_brier <- ggplot(
  performance_table,
  aes(x = reorder(Model, -Brier_Score), y = Brier_Score)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Model Comparison by Brier Score",
    x = "Model",
    y = "Brier Score"
  ) +
  theme_minimal()

ggsave(
  file.path(output_dir, "figures", "ModelComparison_BrierScore.png"),
  p_brier,
  width = 8,
  height = 5,
  dpi = 300
)

###############################################################
# 30. Save Session Information
###############################################################

capture.output(
  sessionInfo(),
  file = file.path(output_dir, "supplementary", "Session_Info.txt")
)

###############################################################
# 31. Completion Message
###############################################################

message("Analysis completed successfully.")
message("All outputs saved in: ", output_dir)
