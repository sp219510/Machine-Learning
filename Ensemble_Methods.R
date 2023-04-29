#------------------------------------------
## STAT 642
## Classification Analysis
## Ensemble Methods
#------------------------------------------
#------------------------------------------
######### Preliminary Code #########
#------------------------------------------
## Clear your workspace
# If your workspace is NOT empty, run:
rm(list=ls())
# to clear it
#------------------------------------------
## Set wd
setwd("C:/Users/chh35/OneDrive - Drexel University/Teaching/Drexel/STAT 642/Course Content/Week 8")
#------------------------------------------
## Install new packages
install.packages("caretEnsemble")

#------------------------------------------
## Load libraries

install.packages("fastAdaboost")
install.packages("randomForest")
install.packages("ipred")
library(caret)
library(caretEnsemble)
library(ipred)
library(fastAdaboost)
library(randomForest)
# training multiple models, custom ensembles

## Load Data
FD <- read.csv(file = "FlightDelays.csv",
               stringsAsFactors = FALSE)

#------------------------------------------

## Data Overview

# The FlightDelays.csv data represents
# information about 2,201 flights 
# originating from airports in the 
# Washington, DC area (BWI, DCA, IAD) 
# and arriving at 3 airports in the 
# New York City, NY area (EWR, JFK and 
# LGA). The airline wants to build
# a classification model to predict
# if a flight will be on time or 
# delayed based on characteristics
# about the flight, including the
# days of the week, day of the month,
# distance of the flight, flight carrier
# company, time of day of the flight, 
# and the weather during the flight. The
# company wants to be able to correctly 
# predict delays and finds it most costly
# when they predict a flight will be
# on-time and it is delayed.

#------------------------------------------
######### Class Code #########
#------------------------------------------

## Data Exploration & Preparation

# First, we can view high-level information
# about the dataframe
str(FD)

## Prepare Target (Y) Variable
FD$delay <- factor(FD$delay)

# We can use the plot() function
# to create a barplot
plot(FD$delay,
     main = "Delay")

## Prepare Predictor (X) Variables
# Based on the data description, we can 
# identify our potential predictor variables. 
# We will set up named vectors so that we 
# can easily refer to the variable groupings 
# and convert them as needed.

## Categorical

# Nominal (Unordered) Factor Variables
noms <- c("carrier", "dest", "origin", "weather")
# We use the lapply() function to convert the
# variables to unordered factor variables
FD[ ,noms] <- lapply(X = FD[ ,noms], 
                     FUN = factor)


# Ordinal (Ordered) Factor Variables
ords <- c("dayweek", "daymonth", "distance")
# We use the lapply() function to convert the
# variables to ordered factor variables
# Since they are integer valued, the default
# ordering will be correct and we can convert
# them all at the same time.
FD[ ,ords] <- lapply(X = FD[ ,ords], 
                     FUN = factor, 
                     ordered = TRUE)

## Numeric
nums <- c("schedtime", "deptime")

# We combine the 3 vectors to create a 
# vector of the names of all of our 
# predictor variables (named vars) that 
# we want to use to predict delay.
vars <- c(noms, ords, nums)

# We can obtain summary informaton for
# our prepared data
summary(FD[,c(vars, "delay")])

#------------------------------------------

## Preprocessing & Transformation

# Since we will be building ensembles of Decision
# Trees, we have the same considerations as
# DT. We know that Decision Trees can handle 
# missing values (we can impute or eliminate 
# up-front or tell the model how to handle them), 
# irrelevant and redundant variables and no
# rescaling/standardization is needed.
# For this reason, we can use the dataset
# as-is in our modeling, without any 
# transformations.

## 1. Missing Values
# If missing values are present, we 
# can remove them (na.omit()), perform
# imputation, or leave them as-is and
# identify how to handle them during DT
# analysis.
any(is.na(FD))

#------------------------------------------

## Training and Testing

# Splitting the data into training and 
# testing sets using an 85/15 split rule

# Initialize random seed
set.seed(831) 

# Create list of training indices
sub <- createDataPartition(y = FD$delay, # target variable
                           p = 0.85, # % in training
                           list = FALSE)

# Subset the transformed data
# to create the training (train)
# and testing (test) datasets
train <- FD[sub, ] # create train dataframe
test <- FD[-sub, ] # create test dataframe
#------------------------------------------

## Ensemble Methods using caret/caretEnsemble

# We will train all 3 models at the same time
# We will use the caretList() function in the 
# caretEnsemble package to train multiple models
# using the same resamples and resampling method.
# As with the train() function, for caretList, we
# will need a trainControl object, which will be
# used as input to the trControl argument. To
# reduce training time, we will perform 5-fold
# cross validation (not repeated).
# Note: parallel processing can be used, using
# parallel and doParallel packages.

ctrl <- trainControl(method = "cv", # k fold cross-validation
                     number = 5, # k = 5 folds
                     savePredictions = "final") # save final resampling summary measures only

# We use the caretList() function and define x,
# y, and trControl as usual. To identify the
# models, hyperparameters to tune (if any) and 
# any arguments to pass to the model (including 
# preProcess). We use the tuneList argument and 
# specify a named list, with models defined using 
# the caretModelSpec() function.

# Hyperparameters:
# Bagging: there are no hyperparameters to tune
# Random Forest: 
# mtry: m, the number of random predictors to
#       use to split on, which defaults to
floor(sqrt(length(vars)))
#       the default grid search will conduct a simple
#       search of values between 2 and p, the number 
#       of predictors. By default, there are 500 trees
# Boosting: 
# nIter: number of classifiers in ensemble
# method: Adaboost.M1 (returns class label) 
#         or Real adaboost (generalization of M1, 
#         returns prob. of class )
#         membership)

set.seed(831) # initialize random seed

ensemble_models <- caretList(x = train[ ,vars], # use vars as predictors
                             y = train$delay, # predict delay variable
                             trControl = ctrl, # control object
                             tuneList = list(
                               bagging = caretModelSpec(method = "treebag"), # use ipred package
                               randforest = caretModelSpec(method = "rf", # use randomForest package
                                                           tuneGrid = expand.grid(mtry = seq(from = 3, # min mtry
                                                                                             to = 9, # max mtry
                                                                                             by = 2))), # count by
                               boosting = caretModelSpec(method = "adaboost", # use fastAdaboost package
                                                         tuneLength = 5) # default grid search 
                             )
)

# To view our optimal hyperparameters for
# the models that we tuned (RF and boosting)
# we can index the caretList object

# Random Forest
ensemble_models[["randforest"]]$results
# Boosting
ensemble_models[["boosting"]]$results

# We can use the resamples() function to
# combines and visualize our resampling
# results. We will save this as results.
results <- resamples(ensemble_models)

# To view summary information for the best
# models for Accuracy and Kappa, we can use
# the summary() function on our resamples 
# object.
summary(results)

# We can visualize the distribution of
# our Accuracy and Kappa across the 3
# ensemble models
bwplot(results) # distribution (box plots)
dotplot(results) # confidence intervals for mean


## Variable Importance
# We can use the varImp() function in the
# caret package and plot() to view variable
# importance (scaled, so that most important 
# = 100) 

# Bagging
plot(varImp(ensemble_models$bagging))
# Random Forest
plot(varImp(ensemble_models$randforest))
# Boosting
plot(varImp(ensemble_models$boosting))

#------------------------------------------

## Model Performance

# For our bagging and random forest models,
# to get an honest estimate of performance
# we would consider OOB predictions on the
# training data. In this case, we will
# use the average cross-validated performance
# above to assess goodness of fit.

## Testing Performance 

# When we use the predict() function with
# a caretList object, it creates a matrix
# object. We save this as a dataframe and
# convert the predictions to factors
ens_preds <- data.frame(predict(object = ensemble_models,
                                newdata = test),
                        stringsAsFactors = TRUE)
# Next, we can obtain confusionMatrix() output
# of each of the models and save the output in
# a list (named ph)
ph <- list()
for (i in 1:ncol(ens_preds)){
  ph[[i]] <- confusionMatrix(data = ens_preds[ ,i],
                             reference = test$delay,
                             positive = "delayed",
                             mode = "everything")
  names(ph)[[i]] <- colnames(ens_preds)[i]
}

## Comparing Performance Across Methods
# We can index the list and use cbind()
# to view our performance side-by-side

## Overall
cbind(bag = ph[["bagging"]]$overall,
      rf = ph[["randforest"]]$overall,
      boost = ph[["boosting"]]$overall)

## By Class
cbind(bag = ph[["bagging"]]$byClass,
      rf = ph[["randforest"]]$byClass,
      boost = ph[["boosting"]]$byClass)

## Goodness of Fit
# We will compare the average Accuracy
# and Kappa for the training results
# to the test performance

## Training
# First, we get the average training 
# performance for the models (colMeans())
# Then, for compatibility with the 
# confusionMatrix output we convert to
# a matrix (matrix()) and transpose (t())
t(matrix(colMeans(results$values[ ,-1]), 
         nrow = 3, 
         byrow = TRUE,
         dimnames = list(c("bag", "rf", "boost"), 
                         c("Accuracy", "Kappa"))))

## Testing
# Next, we can compare the model performance
# on the testing set, focusing on Accuracy
# and Kappa (rows 1 and 2)
cbind(bag = ph[["bagging"]]$overall,
      rf = ph[["randforest"]]$overall,
      boost = ph[["boosting"]]$overall)[1:2,]

