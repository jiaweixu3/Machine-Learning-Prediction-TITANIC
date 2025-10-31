rm(list = ls())
setwd("C:/Users/yojia/OneDrive/TITANICCCCC/LAB2")
setwd("C:/Users/tolmy/OneDrive/Documentos/LAB1/LAB1/LAB2")
load("titanic_train.Rdata")

# Import libraries needed
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
set.seed(1) # Set seed for reproducibility

# Prepare the dataframe
str(titanic.train)

titanic.train$Cabin = factor(titanic.train$Cabin != "")
titanic.train$Ticket = NULL
titanic.train$Survived = factor(titanic.train$Survived, levels = c(0, 1), labels = c("Not survived", "Survived"))

summary(titanic.train)
# There is no NA values so no need to replace values

# Hyper-parameter selection for Decision Tree
depths = 2:10
minsplit_vals = seq(from = 5, to = 50, by = 5)
bucket_vals = seq(from = 5, to = 20, by = 5)
param = expand.grid(depth = depths, minsplit = minsplit_vals, bucket = bucket_vals)


# K-fold cross-validation
k = 5
folds = list()
inivec = 1:length(titanic.train$Survived)
for (i in 1:(k - 1)) {
  split = sample(inivec, floor(length(titanic.train$Survived) / k), replace = FALSE)
  folds[[i]] = split
  inivec = inivec[!(inivec %in% split)]
}
folds[[k]] = inivec

model_quality = data.frame(acc = rep(0, k),
                           sensitivity = rep(0, k),
                           specificity = rep(0, k))
for (j in 1:nrow(param)) {
  print(paste("Model", as.character(j)))
  # Get the result of a model with the parameters j row of param
  # Iterative process of k fold
  for (i in 1:k) {
    # Training set and test set
    training = titanic.train[-folds[[i]], ]
    test = titanic.train[folds[[i]], ]
    
    # Make decision tree model
    DT_model = rpart(formula = Survived ~ ., data = training,
                     method = "class",
                     control = rpart.control(maxdepth = param$depth[j],
                                             minsplit = param$minsplit[j],
                                             minbucket = param$bucket[j]))
    
    # Prediction of the model
    pred = predict(DT_model, test, type = "class")
    
    # Confusion matrix
    Conf_matrix = table(test$Survived, pred, dnn = c("Actual value", "Predicted value"))
    
    # Store results in data frame
    model_quality$acc[i] = sum(diag(Conf_matrix)) / sum(Conf_matrix)
    model_quality$sensitivity[i] = Conf_matrix[1, 1] / sum(Conf_matrix[1, ])
    model_quality$specificity[i] = Conf_matrix[2, 2] / sum(Conf_matrix[2, ])
  }
  # Get the mean of the results of each parameter
  param$accuracy[j] = mean(model_quality$acc)
  param$sensitivity[j] = mean(model_quality$sensitivity)
  param$specificity[j] = mean(model_quality$specificity)
}

#Plot mean accuracy, sensitivity and specificity (meanacc)
#Against depth
ggplot() + aes(x=as.factor(param$depth),
               y = apply(param[, c(4:6)], MARGIN = 1, FUN = mean)) +
                 geom_boxplot() + labs(title="Detph vs meanacc",x="max depth", y="meanacc")
ggsave("graphs/depthvsmeanacc.png", units="cm", width = 16.9, height = 16.9)

#Against minsplit
ggplot() + aes(x=as.factor(param$minsplit),
               y = apply(param[, c(4:6)], MARGIN = 1, FUN = mean)) +
  geom_boxplot() + labs(title="minsplit vs meanacc",x="minsplit", y="meanacc")
ggsave("graphs/minsplitvsmeanacc.png", units="cm", width = 16.9, height = 16.9)

#Against minbucket
ggplot() + aes(x=as.factor(param$bucket),
               y = apply(param[, c(4:6)], MARGIN = 1, FUN = mean)) +
  geom_boxplot() + labs(title="minbucket vs meanacc",x="minbucket", y="meanacc")
ggsave("graphs/minbucketvsmeanacc.png", units="cm", width = 16.9, height = 16.9)



#Select the best parameters for the DT model
best = which.max(apply(param[, c(4:6)], MARGIN = 1, FUN = mean))

best_tree = rpart(formula = Survived ~ ., data = titanic.train,
                   method = "class",
                   control = rpart.control(maxdepth = param$depth[best],
                                           minsplit = param$minsplit[best],
                                           minbucket = param$bucket[best]))
#matrix to compare best DT with RF
compare_matrix = data.frame(name = rep(0,2),
                            acc = rep(0,2),
                            sensitivity = rep(0,2),
                            specificity = rep(0,2))
compare_matrix[1,] = c("DT",param[best,c(4:6)])
summary(best_tree)
plot(best_tree$variable.importance,
     xaxt="n",
     xlab="Variable", ylab="Importance"
)

title("Variable importance")

# Adjust the size (cex) and color (col) parameters in the points function
points(x = seq_along(best_tree$variable.importance),
       y = best_tree$variable.importance,
       pch = 16,  # Code for a solid circle
       cex = 2,   # Adjust the size of the circles
       col = "blue"  # Set the color of the circles to blue
)

axis(side=1, 
     at=seq(1, length(best_tree$variable.importance), by=1),
     labels = names(best_tree$variable.importance),
)

grid()


# Plot the final decision tree model
prp(best_tree,
    type = 2,
    extra = 104,
    nn = TRUE,
    box.col = "aliceblue", shadow.col = "darkgoldenrod2",
    digits = 2,
    roundint = FALSE)



#-----------------------------------------------------------------------------

# Random Forest

#Hyper-parameter selection
ntrees = seq(from=50, to=300, by=25)
mtrys = seq(from=1, to=7, by=1)
param = expand.grid(ntree = ntrees, mtry = mtrys)

model_quality = data.frame(acc = rep(0,k),
                           sensitivity = rep(0,k),
                           specifitivy = rep(0,k))
for (j in 1:nrow(param)){
  print(paste("Model",as.character(j)))
  #Get the result of a model with the parameters j row of param
  #Iterative process of k fold
  for (i in 1:k){
    #Training set and test set
    training = titanic.train[-folds[[i]],]
    test = titanic.train[folds[[i]],]
    
    #Make random forest model
    RF_model = randomForest(formula=Survived~.,data=training,
                            ntree = param$ntree[j], mtry = param$mtry[j])
    
    #Prediction of the model
    pred = predict(RF_model, test, type = "class")
    
    #Confusion matrix
    Conf_matrix = table(test$Survived, pred, dnn=c("Actual value", "Predicted value"))
    
    #Store results in data frame
    model_quality$acc[i] = sum(diag(Conf_matrix)) / sum(Conf_matrix)
    model_quality$sensitivity[i] = Conf_matrix[1,1] / sum(Conf_matrix[1,])
    model_quality$specificity[i] = Conf_matrix[2,2] / sum(Conf_matrix[2,])
  }
  #Get the mean of the results of each parameter
  param$accuracy[j] = mean(model_quality$acc)
  param$sensitivity[j] = mean(model_quality$sensitivity)
  param$specificity[j] = mean(model_quality$specificity)
  
}

#Plot mean accuracy, sensitivity and specificity (meanacc)
#Against ntree
ggplot() + aes(x=as.factor(param$ntree),
              y = apply(param[, c(3:5)], MARGIN = 1, FUN = mean)) +
  geom_boxplot() + labs(title="ntree vs meanacc",x="ntree", y="meanacc")
ggsave("graphs/ntreevsmeanacc.png", units="cm", width = 16.9, height = 16.9)

#Against mtry
ggplot() + aes(x=as.factor(param$mtry),
               y = apply(param[, c(3:5)], MARGIN = 1, FUN = mean)) +
  geom_boxplot() + labs(title="mtry vs meanacc",x="mtry", y="meanacc")
ggsave("graphs/mtryvsmeanacc.png", units="cm", width = 16.9, height = 16.9)


#Select parameters for best RF model
best = which.max(apply(param[,c(3:5)],MARGIN = 1,FUN=mean))
best_forest = randomForest(formula=Survived~.,data = titanic.train,
                          ntree = param$ntree[best], mtry = param$mtry[best])
#Matrix to compare DT with RF
compare_matrix[2,] = c("RF",param[best,c(3:5)])
for (i in 1:nrow(compare_matrix)){
  compare_matrix$meanacc[i] = mean(c(compare_matrix$acc[i], compare_matrix$sensitivity[i], compare_matrix$specificity[i]))
}
compare_matrix$meanacc = as.numeric(compare_matrix$meanacc) 

#-------------------------------------------------------------------------------

#Comparing DT with RF

#1) Using metrics from a trained model with all the data
# Confusion matrix for the best decision tree
pred_tree = predict(best_tree, titanic.train, type = "class")
conf_matrix_tree = table(titanic.train$Survived, pred_tree, dnn = c("Actual value", "Predicted value"))

# Calculate accuracy, sensitivity, and specificity for the decision tree
accuracy_tree = sum(diag(conf_matrix_tree)) / sum(conf_matrix_tree)
sensitivity_tree = conf_matrix_tree[1, 1] / sum(conf_matrix_tree[1, ])
specificity_tree = conf_matrix_tree[2, 2] / sum(conf_matrix_tree[2, ])

# Confusion matrix for the best random forest
pred_rf = predict(best_forest, titanic.train, type = "class")
conf_matrix_rf = table(titanic.train$Survived, pred_rf, dnn = c("Actual value", "Predicted value"))

# Calculate accuracy, sensitivity, and specificity for the random forest
accuracy_rf = sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
sensitivity_rf = conf_matrix_rf[1, 1] / sum(conf_matrix_rf[1, ])
specificity_rf = conf_matrix_rf[2, 2] / sum(conf_matrix_rf[2, ])

# Compare models based on mean accuracy, sensitivity, and specificity
metrics_tree = c(accuracy_tree, sensitivity_tree, specificity_tree)
metrics_rf = c(accuracy_rf, sensitivity_rf, specificity_rf)

cat("\nComparison of Mean Metrics in titanic.train:\n")
if (mean(metrics_tree) > mean(metrics_rf)) {
  cat("The Decision Tree model is better.\n")
} else if (mean(metrics_rf) > mean(metrics_tree)) {
  cat("The Random Forest model is better.\n")
} else {
  cat("Both models have similar performance.\n")
}

#2) Using metrics from after selecting models
metrics_tree_2 = compare_matrix$meanacc[1]
metrics_rf_2 = compare_matrix$meanacc[2]


cat("\nComparison of Mean Metrics from after hyperparameter selection:\n")
if (metrics_tree_2 > metrics_rf_2) {
  cat("The Decision Tree model is better.\n")
} else if (metrics_rf_2 > metrics_tree_2) {
  cat("The Random Forest model is better.\n")
} else {
  cat("Both models have similar performance.\n")
}




#--------------------------------------------------------------

bestclassifier = randomForest(formula=Survived~.,data = titanic.train,
                              ntree = 225, mtry = 3)

my_model = function(test_set){
  # We have to make the same preprocessing to my test set storaged in variable test_set
  test_set$Cabin = factor(test_set$Cabin != "")
  test_set$Ticket = NULL
  test_set$Survived = factor(test_set$Survived, levels = c(0, 1), labels = c("Survived", "Not Survived"))
  
  # Use the best classifier to forecast survival
  pred = predict(bestclassifier,test_set,type="class")
  # Compute the confusion matrix
  conf_matrix = table(test_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))
  conf_matrix_prop = prop.table(conf_matrix)
  
  # Compute error estimates
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  precision = conf_matrix[1,1]/sum(conf_matrix[,1])
  specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
  return(list(prediction=pred,
              conf_matrix=conf_matrix,
              conf_matrix_prop=conf_matrix_prop,
              accuracy=accuracy,
              precision=precision,
              specificity=specificity))
}

save(bestclassifier,my_model,file="my_model.RData")
