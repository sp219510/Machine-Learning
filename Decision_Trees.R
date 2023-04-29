#------------------------------------------
## STAT 642
## Classification Analysis
## Decision Trees 
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
setwd("C:/Users/chh35/OneDrive - Drexel University/Teaching/Drexel/STAT 642/Course Content/Week 6")
#------------------------------------------
## Install new packages
install.packages(c("rpart", # basic decision tree
                   "rpart.plot")) # decision tree plotting


#------------------------------------------
## Load libraries
library(caret)
library(rpart)
library(rpart.plot)

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
noms <- c("carrier", "dest", "origin", "weather", "dayweek")
# We use the lapply() function to convert the
# variables to unordered factor variables
FD[ ,noms] <- lapply(X = FD[ ,noms], 
                     FUN = factor)


# Ordinal (Ordered) Factor Variables
ords <- c("distance")
FD$distance <- factor(x = FD$distance,
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
######### Decision Trees #########
#------------------------------------------

## Preprocessing & Transformation

# We know that Decision Trees can handle 
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

## Analysis

### 1. Basic Model (rpart() in the rpart package)
# We use the rpart() function in the rpart 
# package to perform basic DT classification 
# on our training dataset.

# If we have NA values that we did not
# handle during preprocessing, we can
# use the na.action argument, which defaults
# to na.rpart, which removes observations
# with NA values for y but keeps observations
# with NA values for predictor variables.

# Note: rpart defaults to using Gini for 
# split decisions but can change to using 
# entropy by specifying split = "information"
# in parms = list(). cp defaults to 0.01.
FD.rpart <- rpart(formula = delay ~ ., # Y ~ all other variables in dataframe
                  data = train[ ,c(vars, "delay")], # include only relevant variables
                  method = "class")

# We can see the basic output of our
# Decision Tree model
FD.rpart

# We can use the variable.importance
# component of the rpart object to 
# obtain variable importance
FD.rpart$variable.importance

## Tree Plots
# We can use either the prp() function 
# or the rpart.plot() function in the 
# rpart.plot package to plot our 
# rpart object (FD.rpart).
prp(x = FD.rpart, # rpart object
    extra = 2) # include proportion of correct predictions

## Training Performance
# We use the predict() function to generate 
# class predictions for our training set
base.trpreds <- predict(object = FD.rpart, # DT model
                        newdata = train, # training data
                        type = "class") # class predictions

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# training dataset (train).
DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train$delay, # actual
                                 positive = "delayed",
                                 mode = "everything")
DT_train_conf


## Testing Performance
# We use the predict() function to generate 
# class predictions for our testing set
base.tepreds <- predict(object = FD.rpart, # DT model
                        newdata = test, # testing data
                        type = "class")

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# testing dataset (test).
DT_test_conf <- confusionMatrix(data = base.tepreds, # predictions
                                reference = test$delay, # actual
                                positive = "delayed",
                                mode = "everything")
DT_test_conf


## Goodness of Fit

# To assess if the model is balanced,
# underfitting or overfitting, we compare
# the performance on the training and
# testing. We can use the cbind() function
# to compare side-by-side.

# Overall
cbind(Training = DT_train_conf$overall,
      Testing = DT_test_conf$overall)

# Class-Level
cbind(Training = DT_train_conf$byClass,
      Testing = DT_test_conf$byClass)

#------------------------------------------

### 2. Hyperparameter Tuning Model

# Using the train() function in the caret 
# package

# We will perform a grid search for the 
# optimal cp value.

# We want to tune the cost complexity 
# parameter, or cp. We choose the cp 
# that is associated with the smallest 
# cross-validated error (highest accuracy)

# We will search over a grid of values
# from 0 to 0.05. We use the expand.grid()
# function to define the search space
grids <- expand.grid(cp = seq(from = 0,
                              to = 0.05,
                              by = 0.005))
grids

# First, we set up a trainControl object
# (named ctrl) using the trainControl() 
# function in the caret package. We specify 
# that we want to perform 10-fold cross 
# validation, repeated 3 times and specify
# search = "grid" for a grid search. We use 
# this object as input to the trControl 
# argument in the train() function below.
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     search = "grid")


# Next, we initialize a random seed for 
# our cross validation
set.seed(831)

# Then, we use the train() function to
# train the DT model using 5-Fold Cross 
# Validation (repeated 3 times). We set
# tuneGrid equal to our grid search
# objects, grids.
DTFit <- train(form = delay ~ ., # use all variables in data to predict delay
               data = train[ ,c(vars, "delay")], # include only relevant variables
               method = "rpart", # use the rpart package
               trControl = ctrl, # control object
               tuneGrid = grids) # custom grid object for search

# We can view the results of our
# cross validation across cp values
# for Accuracy and Kappa. The output
# will also identify the optimal cp.
DTFit # all results

DTFit$results[DTFit$results$cp %in% DTFit$bestTune,] #best result


# We can plot the cp value vs. Accuracy
plot(DTFit)

# We can view the confusion matrix showing
# the average performance of the model
# across resamples
confusionMatrix(DTFit)

# Decision Trees give us information 
# about Variable Importance. We can use
# the best fit object from the caret
# package to obtain variable importance
# information 
DTFit$finalModel$variable.importance


## Tuned Model Performance

### Training Performance
# We use the predict() function to generate 
# class predictions for our training data set
tune.trpreds <- predict(object = DTFit,
                        newdata = train,
                        type = "prob")

# We use the confusionMatrix() function
# from the caret package
DT_trtune_conf <- confusionMatrix(data = tune.trpreds, # predictions
                                  reference = train$delay, # actual
                                  positive = "delayed",
                                  mode = "everything")
DT_trtune_conf


## Testing Performance
# We use the predict() function to generate class 
# predictions for our testing data set
tune.tepreds <- predict(object = DTFit,
                        newdata = test)

# We use the confusionMatrix() function
# from the caret package
DT_tetune_conf <- confusionMatrix(data = tune.tepreds, # predictions
                                  reference = test$delay, # actual
                                  positive = "delayed",
                                  mode = "everything")
DT_tetune_conf


## Goodness of Fit
# To assess if the model is balanced,
# underfitting or overfitting, we compare
# the performance on the training and
# testing. We can use the cbind() function
# to compare side-by-side.

# Overall
cbind(Training = DT_trtune_conf$overall,
      Testing = DT_tetune_conf$overall)

# Class-Level
cbind(Training = DT_trtune_conf$byClass,
      Testing = DT_tetune_conf$byClass)


