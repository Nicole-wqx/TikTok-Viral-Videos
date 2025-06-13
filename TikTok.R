setwd("/Users/wqx/Desktop/Machine Learning/HW/Final Project")
tt <- read.csv("tiktok_sample_25.csv", header = TRUE)
library(ggplot2)
library(dplyr)
library(tidyr)
library(splines)
library(car)
library(psych)
library(MASS)
library(glmnet)
library(mgcv)
library(rpart)
library(rpart.plot)

# Preprocessing
tt <- tt[!is.na (tt$Emotion_SURPRISED), ]

hist(tt$Share.Count, breaks = 50, main = "Share Count Distribution", xlab = "Share Count")
hist(tt$View.Count, breaks = 50, main = "View Count Distribution", xlab = "View Count")
hist(tt$Comment.Count, breaks = 50, main = "Comment Count Distribution", xlab = "Comment Count")
hist(tt$Like.Count, breaks = 50, main = "Like Count Distribution", xlab = "Like Count")

tt$gender.deepface <- as.factor(tt$gender.deepface)
tt$race <- as.factor(tt$race)
tt$gender.amazon <- as.factor(tt$gender.amazon)
tt$Smile <- as.factor(tt$Smile)
tt$Verified.Status <- as.factor(tt$Verified.Status)
tt$log_share_count <- log(tt$Share.Count + 1)
tt$log_view_count <- log(tt$View.Count + 1)
tt$log_like_count <- log(tt$Like.Count + 1)
tt$log_comment_count <- log(tt$Comment.Count + 1)
tt$log_follower_count <- log(tt$Follower.Count + 1)
tt$log_following_count <- log(tt$Following.Count + 1)
tt$log_likesum_count <- log(tt$Likes.sum.Count + 1)



# LASSO Regression for feature selection
set.seed(12345) 
tt <- na.omit(tt)
train_idx <- sample(1:nrow(tt), 0.7 * nrow(tt))
train_data <- tt[train_idx, ]
test_data <- tt[-train_idx, ]

fit1 = lm(log_comment_count~
            log_share_count+log_view_count+log_like_count+
            Emotion_SURPRISED+Emotion_FEAR+Emotion_CONFUSED+Emotion_ANGRY+Emotion_SAD+Emotion_DISGUSTED+Emotion_HAPPY+Emotion_CALM+
            transcript_Anger + transcript_Disgust + transcript_Fear + transcript_Joy + transcript_Sadness + transcript_Surprise +
            log_following_count+log_follower_count+as.factor(Verified.Status)+log_likesum_count, data = tt)
summary(fit1)
vif(fit1)
drop1(fit1)

tt$Verified.Status <- as.numeric(tt$Verified.Status)
tt$Verified.Status <- as.factor(tt$Verified.Status)
colnames(tt)
preds <- as.matrix(tt[, c("median_negative_sentiment",
                          "log_share_count", "log_view_count", "log_like_count", 
                          "Emotion_SURPRISED", "Emotion_FEAR", "Emotion_CONFUSED", "Emotion_ANGRY",
                          "Emotion_SAD", "Emotion_DISGUSTED", "Emotion_HAPPY", "Emotion_CALM",
                          "log_following_count", "log_follower_count", "Verified.Status","log_likesum_count",
                          "transcript_Anger", "transcript_Disgust", "transcript_Fear",
                          "transcript_Joy", "transcript_Sadness", "transcript_Surprise")])
outcome <- tt$log_comment_count

set.seed(12345)
cv_lasso <- cv.glmnet(preds, outcome, alpha = 1)
print(cv_lasso)
best_lambda_lasso <- cv_lasso$lambda.min

final_lasso <- glmnet(preds, outcome, alpha = 1, lambda = best_lambda_lasso)
coefficients_lasso <- coef(final_lasso, s = best_lambda_lasso)
print(coefficients_lasso)
## emotion_SURPRISED, Emotion_CONFUSED, trancript_Joy are dropped 


#Random Forest
set.seed(12345) 

tt <- na.omit(tt)
tt$Verified.Status <- as.factor(tt$Verified.Status)
train_idx <- sample(1:nrow(tt), 0.7 * nrow(tt))
train_data <- tt[train_idx, ]
test_data <- tt[-train_idx, ]

library(randomForest)

selected_vars <- c(
  "log_comment_count",
  "log_share_count", "log_view_count", "log_like_count", 
  "log_following_count", "log_follower_count", "Verified.Status", "log_likesum_count",
  "median_negative_sentiment",
  "Emotion_FEAR", "Emotion_ANGRY", "Emotion_SAD", "Emotion_DISGUSTED", "Emotion_HAPPY", "Emotion_CALM",
  "transcript_Anger", "transcript_Disgust", "transcript_Fear", "transcript_Sadness", "transcript_Surprise"
)

train_subset <- train_data[, selected_vars]
test_subset <- test_data[, selected_vars]

rf_model <- randomForest(
  log_comment_count ~ .,  
  data = train_subset,
  ntree = 500,
  mtry = floor(sqrt(length(selected_vars) - 1)),  
  importance = TRUE,
  na.action = na.omit
)
print(rf_model)

oob_error <- rf_model$err.rate
plot(rf_model, main = "OOB Error vs. Number of Trees")

oob_pred <- rf_model$predicted
actual <- rf_model$y

rmse <- sqrt(mean((actual - oob_pred)^2))
cat("OOB RMSE:", rmse, "\n")

rsq <- 1 - sum((actual - oob_pred)^2) / sum((actual - mean(actual))^2)
cat("OOB R-squared:", rsq, "\n")

# Predict on test data 
predicted_values2 <- predict(rf_model, newdata = test_data)
actual_values2 <- test_data$log_comment_count

# Calculate RMSE
rmse_test2 <- sqrt(mean((predicted_values2 - actual_values2)^2))
print(paste("Test RMSE:", round(rmse_test2, 4)))

# Calculate R-squared
sst2 <- sum((actual_values2 - mean(actual_values2))^2)
sse2 <- sum((predicted_values2 - actual_values2)^2)
r_squared_test2 <- 1 - sse2/sst2
print(paste("Test R-squared:", round(r_squared_test2, 4)))


#Tune the RF model
p <- ncol(train_subset) - 1  
tune_grid <- expand.grid(mtry = c(floor(p/4), floor(p/3), floor(p/2), floor(p)))

library(caret)
rf_tuned <- train(
  log_comment_count ~ .,
  data = train_subset,
  method = "rf",
  tuneGrid = tune_grid,
  trControl = trainControl(method = "cv", number = 5),  
  ntree = 300
)

print(rf_tuned$bestTune)
plot(rf_tuned)

final_rf <- randomForest(
  log_comment_count ~ .,
  data = train_subset,
  # ntree = rf_tuned$bestTune$ntree,
  mtry = rf_tuned$bestTune$mtry,
  importance = TRUE
)

print(final_rf)

oob_pred2 <- final_rf$predicted
actual2 <- final_rf$y

rmse2 <- sqrt(mean((actual2 - oob_pred2)^2))
cat("OOB RMSE:", rmse2, "\n")

rsq2 <- 1 - sum((actual2 - oob_pred2)^2) / sum((actual2 - mean(actual2))^2)
cat("OOB R-squared:", rsq2, "\n")

predicted_values <- predict(final_rf, newdata = test_data)
actual_values <- test_data$log_comment_count

rmse_test <- sqrt(mean((predicted_values - actual_values)^2))
print(paste("Test RMSE:", round(rmse_test, 4)))

sst <- sum((actual_values - mean(actual_values))^2)
sse <- sum((predicted_values - actual_values)^2)
r_squared_test <- 1 - sse/sst
print(paste("Test R-squared:", round(r_squared_test, 4)))

varImpPlot(final_rf, main = "Variable Importance")

# XGBoost to predict viral videos
library(xgboost)
range(tt$log_comment_count)
tt$viral <- ifelse(tt$log_comment_count > 5, 1, 0)  # Define viral videos as those with log_view_count > 10
set.seed(12345)
train_idx <- sample(1:nrow(tt), 0.7 * nrow(tt))
train_data <- tt[train_idx, ]
test_data <- tt[-train_idx, ]
selected_vars_xg <- c(
  "viral","age","gender.deepface","race",
  "log_share_count", "log_view_count", "log_like_count", 
  "log_following_count", "log_follower_count", "Verified.Status", "log_likesum_count"
)
train_subset <- train_data[, selected_vars_xg]
test_subset <- test_data[, selected_vars_xg]
train_x <- model.matrix(viral ~ .-1, data = train_subset)
train_y <- as.integer(train_subset$viral) - 1
test_x <- model.matrix(viral ~ . - 1, data = test_subset)

viral_levels <- levels(factor(train_subset$viral))
train_y <- as.integer(factor(train_subset$viral, levels = viral_levels)) - 1
test_y  <- as.integer(factor(test_subset$viral, levels = viral_levels)) - 1

xgb_model <- xgboost(
  data = train_x, 
  label = train_y, 
  nrounds = 100, 
  objective = "multi:softmax", 
  num_class = length(unique(train_y)), 
  eta = 0.1,     
  max_depth = 6  
)

install.packages("DiagrammeR")
xgb.plot.tree(model = xgb_model, trees = 0)


pred <- predict(xgb_model, test_x)
train_subset$viral <- factor(train_subset$viral)
pred <- factor(pred, labels = levels(train_subset$viral))
test_subset$viral <- factor(test_subset$viral)

confusionMatrix(pred, test_subset$viral)
accuracy_xg <- sum(pred == test_subset$viral) / length(pred)
print(paste("XGBoost Accuracy:", round(accuracy_xg, 4)))

importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix = importance)

library(pdp)

pdp_plot <- partial(xgb_model, pred.var = "gender.deepface", train = train_x, type = "regression")
plot(pdp_plot)
