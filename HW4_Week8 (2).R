#------------------------------------------
#------------------------------------------
############## Homework # 4 ##############
#------------------------------------------
#------------------------------------------

# Directions: In this assignment, you will use the 
# the Bank1.csv data file to perform ANN and SVM
# Classification Analysis. 
# If an answer requires a written response, use '##' 
# before your answer. You will submit (1) .R 
# R Script file and (1) .RData file, which contains 
# all objects in your workspace created while 
# completing the homework assignment. If you have 
# extra objects in your workspace, clear your 
# workspace, run your HW file and then save your
# workspace using save.image(). 

# Note: Your files (.R and .RData) should be named 
# using the following format: HW#_Group. For instance, 
# the 3rd HW for Group 15 would be: HW3_Group15.
#------------------------------------------

# Data Description

# The Bank1.csv data contains records for 5000
# customers of a bank. The variables contain 
# demographic information (Age, Income, ZIP.Code,
# Family, Education) and banking relationship
# information (CCAvg, Mortgage, SecuritiesAccount,
# CDAccount, Online, CreditCard). The PersonalLoan
# variable is of particular interest, since it
# identifies if the customer responded to their
# marketing campaign for personal loans in the 
# past. The bank would like to predict customers
# that will respond to future Personal Loan
# marketing campaigns using their available
# historical data to perform classification 
# analysis.

# Variable Descriptions:
# ID: unique customer identification
# Age: customer's age (in years)
# Experience: years of professional experience
# Income: customer's annual income (in $1,000s)
# ZIP.Code: zip code of customer's residence
# Family: customer's family size
# CCAvg: average credit card spending per month
#        (in $1,000s)
# Education: Education level (1-Undergrad,
#            2-Graduate, 3-Advanced)
# Mortgage: value of home mortgage (if applicable,
#           in $1,000s)
# PersonalLoan: if the customer accepted a personal
#               loan in the previous marketing campaign
#               (1) or not (0)
# SecuritiesAccount: if the customer has a securities
#                    account with the bank (1) or not (0)
# CDAccount: if the customer has a CD account with
#            the bank (1) or not (0)
# Online: if the customer has online banking setup
#         for use (1) or not (0)
# CreditCard: if the customer has a credit card with
#             the bank (1) or not (0)

#------------------------------------------

## Preliminary

## Clear your workspace
# If your workspace is NOT empty, run:
rm(list=ls())


## 0a. Set your working directory to the location
# of the Bank1.csv file.
setwd("C:/Users/RAJIV KUMAR/Desktop/")

## 0b. Load the caret package.
library(caret) # Load caret libraries
library(Rcpp)
library(ranger)
library(e1071) # SVM
library(NeuralNetTools) # ANN Plot
library(MLeval) # ROC Curve Plot

## 0c. Import the Bank1.csv data as a dataframe
# named B1.
B1 <- read.csv(file = "Bank1.csv")


## 0d. You will use all variables as input to
# predict PersonalLoan EXCEPT ID and ZIP.Code.
# Create a vector named vars that contains
# the names of the predictor variables.
facs <- c("ZIP.code", "ID","SecuritiesAccount", "CDAccount", "Online", "CreditCard","PersonalLoan")
B1$SecuritiesAccount<-factor(x=B1$SecuritiesAccount)
B1$CDAccount<-factor(x=B1$CDAccount)
B1$Online<-factor(x=B1$Online)
B1$CreditCard<-factor(x=B1$CreditCard)
B1$PersonalLoan<-factor(x=B1$PersonalLoan)
ords <- c("Education","Family")

nums <- c("Experience","Age","Income","Mortgage","CCAvg")


vars <- c(facs[-1][-1][-5], ords, nums) #Removed ID and ZIP.Code

#------------------------------------------
############### Part 1: ANN ###############
#------------------------------------------

# 1a. (1) Should you convert any variables to 
# factor variables? If so, make the conversion(s).

B1[,ords] <- lapply(X = B1[,ords], FUN = factor, ordered = TRUE)

# 1b. (2) Prior to performing ANN classification, 
# what preprocessing and transformation steps
# are needed? Perform all necessary steps
# identified. If normalization is needed,
# perform this step during model training.
B1$Education=as.numeric(B1$Education)
B1$Family=as.numeric(B1$Family)
any(is.na(x=B1)) #Check Missing Values

## We found out some missing values, hence removing incomplete rows in the dataset

B1[!complete.cases(B1), ]
na_rows <- rownames(B1)[!complete.cases(B1)]
na_rows
B1$Experience[is.na(B1$Experience)] <- mean(B1$Experience, 
                                            na.rm = TRUE)
cbind(B1[na_rows, "Income"], B1[na_rows, "Income"])

B1_noNA <- na.omit(object = B1) #Using omit function to remove NA values
summary(object = B1_noNA) #checking the summary to see that all missing data has been removed.

##Checking outliers
comp_box<-boxplot(x=B1$CCAvg,main="CCAvg")
outs <- sapply(B1$CCAvg, function(x) which(abs(scale(x)) > 3))
outs
outs <- outs[lapply(outs, length) > 0] 
outs

## All nominal variables are already binary.There is outliers in the CCAvg variable
##which is our continous variable.
## We will normalize the numeric variables when training the model


#------------------------------------------

# 2. (2) Split your data into training (named 
# train) and testing (named test) sets, 
# preserving the distribution of the target 
# variable. Use a 75/25 split ratio and use 
# 673 as your initial seed.

set.seed(673) # initialize the random seed
sub <- createDataPartition(y = B1$PersonalLoan, 
                           p = 0.75, 
                           list = FALSE)

# Creating train and test sets
train <- B1[sub, ] 
test <- B1[-sub, ]

#------------------------------------------

# 3. You will perform hyperparameter tuning to
# find the optimal size and decay values using 
# 5-Fold Cross Validation repeated 3 times. You 
# will use a random search to try 5 random 
# combinations of values. You will use Accuracy
# as the performance measure to identify your 
# optimal hyperparameters.

grids <-  expand.grid(size = seq(from = 3, # min node value
                                 to = 9, # max node value
                                 by = 2), # counting by
                      decay = seq(from = 0, # min wd value
                                  to = 0.1, # max wd value
                                  by = 0.01)) # counting by

grids #Setting up the expand.grid() function for the size and decay hyperparameters

# 3a. (1)  Set up your 5-Fold Cross Validation 
# (repeated 3 times) random search hyperparameter
# tuning model.

ctrl <- trainControl(method = "repeatedcv",
                     number = 5, # 5 folds
                     repeats = 3, # 3 repeats
                     search = "grid") # grid search


# 3b. (2) Initialize 927 as your random seed, 
# then train the ANN classification model with
# hyperparameter tuning using a random search
# of 5 combinations of hyperparameter values. 
# Perform any normalization while training.
# Hint: use your vars vector.

set.seed(927)

#Using the preProcess argument for range (min-max) normalization

annMod <- train(form = PersonalLoan ~., # use all other variables to predict PersonalLoan
                data = train[,-(13:14)], # training data
                preProcess = "range", # apply min-max normalization
                method = "nnet", # use nnet()
                trControl = ctrl, 
                maxit = 200, # increase # of iterations from default (100)
                tuneGrid = grids, # search over the created grid
                trace = FALSE) # suppress output


# 3c. (1) What is the optimal number of 
# nodes in the hidden layer and weight 
# decay value? What are the average accuracy 
# and kappa values associated with the
# optimal hyperparameters?
annMod$bestTune
annMod

##Accuracy was used to select the optimal model using the largest value.
##The final values used for the model were size = 7 and decay = 0.08 and the kappa
##value is 0.8838237 .

#------------------------------------------

# 4a. (2) Obtain predictions and performance 
# information for the training set. Use the 
# class with the smallest number of observations
# as the positive class and obtain all available
# performance information in your output.

tune.tr.preds <- predict(object = annMod, # tuned model
                         newdata = train) # training data



tune_tr_conf <- confusionMatrix(data = tune.tr.preds, # predictions
                                reference = train$PersonalLoan, # actual
                                positive = "1",
                                mode = "everything")

tune_tr_conf

# 4b. (2) Obtain predictions and performance 
# information for the testing set. Use the 
# class with the smallest number of observations
# as the positive class and obtain all available
# performance information in your output.


tune.te.preds <- predict(object = annMod, # tuned model
                         newdata = test) # training data

tune_te_conf <- confusionMatrix(data = tune.te.preds, # predictions
                                reference = test$PersonalLoan, # actual
                                positive = "1",
                                mode = "everything")
tune_te_conf
#------------------------------------------

# 5. (2) Describe the performance and fit
# of the model. Is this a good model? Explain.

# Overall
cbind(Training = tune_tr_conf$overall,
      Testing = tune_te_conf$overall)

# Class-Level
cbind(Training = tune_tr_conf$byClass,
      Testing = tune_te_conf$byClass)

#------------------------------------------
######### Part 2: SVM ########
#------------------------------------------

## Next, you will perform Support Vector
# Machines classification analysis with
# radial kernel. You will use the same
# train/test sets created in Question 2
# when training the model.

# 6a. (2) Prior to performing a SVM classification, 
# analysis what  
# steps are needed? Which of these preprocessing 
# and transformation steps were already performed
# in Question 1b?

# For preprocessing and transformation: missing 
# values need to be handled, categorical 
# variables need to be transformed, outliers 
# and skew should be evaluated and numerical
# variables should be normalized.


## 1. We have already identified missing values and removed them.
## 2. All nominal variables are already binary.
## 3. We will standardize the numeric variables when training the model


# 6b. (2) Can SVM handle redundant variables? If not,
# use the B1 dataframe to identify any redundant
# variables (using a cutoff of .8) and remove them 
# from your vars vector.

##Yes, SVM can handle  Irrelevant and Redundant Variables.

#------------------------------------------

# 7. You will perform hyperparameter tuning to
# find the optimal cost and gamma (called sigma)
# values. You will use a grid search to perform
# 5-fold cross validation, repeated 3 times. 
# You will use Accuracy as the performance measure.
sigma <- trainControl(method = "repeatedcv",
                      number = 5, # k = 5
                      repeats = 3, # repeat 3 times
                      search = "random", # random search
                      classProbs = TRUE, # needed for AUC
                      savePredictions = TRUE, # save the predictions to plot
                      summaryFunction = twoClassSummary)

# 7a. (1) First, set up the grid for your grid
# search. You will search over cost (C) values
# from 1 to 5, counting by 1.
# You will search over sigma values from 0.01 to
# 0.11, counting by 0.05.

grids_svm <-  expand.grid(size = seq(from = 1, # min node value
                                     to = 5, # max node value
                                     by = 1), # counting by
                          decay = seq(from = 0.01, # min wd value
                                      to = 0.11, # max wd value
                                      by = 0.05)) # counting by
grids_svm


# 7b. (1)  Set up your 5-Fold Cross Validation 
# (repeated 3 times) grid search hyperparameter
# tuning model. 

ctrl_svm <- trainControl(method = "repeatedcv",
                     number = 5, # k = 5
                     repeats = 3, # repeat 3 times
                     search = "random", # random search
                     classProbs = TRUE, # needed for AUC
                     savePredictions = TRUE, # save the predictions to plot
                     summaryFunction = twoClassSummary) # for AUC instead of Accuracy & Kappa



# 7c. (2) Initialize 280 as your random seed, then
# perform the SVM classification with
# hyperparameter tuning. Perform any normalization
# while training.
# Hint: use your vars vector.

set.seed(280)

#Using the preProcess argument for range (min-max) normalization

lin_mod <- svm(formula = PersonalLoan ~ ., # use all other variables to predict PersonalLoan
               data = train, # use train data
               method = "C-classification", # classification
               kernel = "linear", # linear kernel
               scale = TRUE) # standardize the data

summary(lin_mod)

# 7d. (1) What are the optimal Cost and 
# sigma values? What are the average accuracy
# and kappa values associated with the
# optimal hyperparameters?

SVMFit <- train(form = PersonalLoan ~ ., # use all other variables to predict PersonalLoan
                data = train, # train using train dataframe
                method = "svmRadial", # radial SVM using kernlab
                preProcess = c("center", "scale"), # standardize
                trControl = ctrl_svm, # sets up search and resampling
                tuneLength = 10, # try 10 random cost values
                metric = "ROC") # use ROC/AUC to choose the best model
SVMFit


# 7e. (+1 EC) Based on the optimal value of 
# the Cost hyperparameter, is your model
# focusing on minimizing training error 
# or maximizing the margin? Explain.


#------------------------------------------

# 8a. (2) Obtain predictions and performance 
# information for the training set. Use the 
# class with the smallest number of observations
# as the positive class all available
# performance information in your output.

tune.tr.preds <- predict(object = SVMFit,
                         newdata = train)
SVM_trtune_conf <- confusionMatrix(data = tune.tr.preds, # predictions
                                   reference = train$PersonalLoan, # actual
                                   positive = "1",
                                   mode = "everything")


# 8b. (2) Obtain predictions and performance 
# information for the testing set. Use the 
# class with the smallest number of observations
# as the positive class and obtain all available
# performance information in your output.


tune.te.preds <- predict(object = SVMFit,
                         newdata = test)
SVM_tetune_conf <- confusionMatrix(data = tune.te.preds, # predictions
                                   reference = test$PersonalLoan, # actual
                                   positive = "1",
                                   mode = "everything")
SVM_tetune_conf
#------------------------------------------

# 9. (2) Describe the performance and fit
# of the model. Is this a good model? Explain.

# Overall
cbind(Training = SVM_trtune_conf$overall,
      Testing = SVM_tetune_conf$overall)

# Class-Level
cbind(Training = SVM_trtune_conf$byClass,
      Testing = SVM_tetune_conf$byClass)

#------------------------------------------

# 10. (+1 EC) Compare the ANN and SVM models. Which
# model would you recommend that the bank use
# to predict response to their Personal Loan
# marketing campaign? Why? Explain.



#------------------------------------------