#------------------------------------------
## STAT 642
## Classification Analysis
## Artificial Neural Networks
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
setwd("C:/Users/chh35/OneDrive - Drexel University/Teaching/Drexel/STAT 642/Course Content/Week 7")
#------------------------------------------
## Install new packages
install.packages("NeuralNetTools")

#------------------------------------------
## Load libraries
library(caret)
library(NeuralNetTools) # ANN Plot


## Load Data
OJ <- read.csv(file = "OJ.csv",
               stringsAsFactors = FALSE)

#------------------------------------------

## Data Overview

# The OJ dataset contains purchase information
# for orange juice purchases made at 5 different
# stores. Customer and product information is
# included, and a large grocery store chain wants
# to use the data to be able to predict if a 
# customer will purchase Citrus Hill (CH) or 
# Minute Maid (MM) orange juice. The grocery
# store prioritizes being able to correctly predict
# Citrus Hill purchases

# Variable Descriptions:
# Purchase: OJ purchase choice, either Citrus
#           Hill (CH) or Minute Maid (MM)
# Week of Purchase: number identifying the week
#                   of purchase
# Store ID: the identification of the store of
#           OJ purchase 
# PriceCH: The price charged for CH
# PriceMM: The price charged for MM
# DiscCH: Discount offered by CH
# DiscMM: Discount offered by MM
# SpecialCH: Indicates if there was a promotion
#            for CH (1) or not (0)
# SpecialMM: Indicates if there was a promotion
#            for MM (1) or not (0)
# LoyalCH: customer brand loyalty to CH
# SalePriceMM: The sale price of MM
# SalePriceCH: The sale price of CH
# PriceDiff: MM Sale Price - CH Sale Price
# PctDiscMM: Percentage discount for MM
# PctDiscCH: Percentage discount for CH
# ListPriceDiff: MM List Price - CH List Price

#------------------------------------------
######### Class Code #########
#------------------------------------------

## Data Exploration & Preparation

# First, we can view high-level information
# about the OJ dataframe
str(OJ)

## Prepare Target (Y) Variable
OJ$Purchase <- factor(OJ$Purchase)

# We can use the plot() function
# to create a barplot
plot(OJ$Purchase,
     main = "OJ Purchase Type")

## Prepare Predictor (X) Variables
# Based on the data description, we can 
# identify our potential predictor variables. 

## Categorical

# Nominal (Unordered) Factor Variables
# Our nominal variables are StoreID, SpecialCH,
# SpecialMM

# Since SpecialCH and SpecialMM are already
# binary, we can leave them as-is, but we will
# convert StoreID to a factor (for now).
OJ$StoreID <- factor(OJ$StoreID)


# Ordinal (Ordered) Factor Variables
# We do not have any ordinal variables

## Numeric
# All other variables are numeric
# We can obtain summary information for
# our prepared data
summary(OJ)

#------------------------------------------

## Data Preprocessing & Transformation

# ANN can handle redundant variables, but 
# missing values need to be handled, categorical 
# variables need to be binarized and rescaling 
# should be done

## 1. Missing values
# We check for missing values. If
# missing values are present, we can
# either remove them row-wise or perform
# imputation.
any(is.na(OJ))

## 2. Transform Categorical Variables

#### Nominal Variables
# 2 class levels: binarize using the 
##  class2ind() function from the caret package  
# > 2 class levels: binarize using the 
##  dummyVars() function from the caret 
##  package and the predict() function 

#### Ordinal variables:
# To preserve class level ordering, convert
# to numeric using as.numeric() function

# We need to binarize the StoreID variable.
cats <- dummyVars(formula =  ~ StoreID,
                  data = OJ)
cats_dums <- predict(object = cats, 
                     newdata = OJ)

# Combine binarized variables (cats_dum) with data
# (OJ), excluding the StoreID factor variable
OJ_dum <- data.frame(OJ[ ,!names(OJ) %in% "StoreID"],
                     cats_dums)

## 3. Evaluate Outliers
# We will evaluate outliers on our continuous
# numerical variables (PriceCH, PriceMM, LoyalCH,
# PctDiscMM, PctDiscCH)
outs <- sapply(OJ_dum[,-c(1:2, 7:8, 16:20)], function(x) which(abs(scale(x)) > 3))
outs

# Let's take a closter look at the variables 
# with outliers
outs <- outs[lapply(outs, length) > 0] 

# Plot histograms
par(mfrow = c(3,3))
for (i in 1:length(outs)){
  hist(OJ_dum[,names(outs)[i]],
       main = names(outs)[i],
       xlab = "")
}
par(mfrow = c(1,1)) # return to default plot window

# The Disc and PctDisc variables
# are severely right skewed. We can
# omit the PctDisc variables as
# predictors and transform the Disc 
# variables to binary indicators

OJ_dum[,c("DiscCH", "DiscMM")] <-
  ifelse(test = OJ_dum[,c("DiscCH", "DiscMM")] > 0,
         yes = 1,
         no = 0)

## 4. Normalize Numeric Variables
# We will use min-max normalization to
# rescale our numeric variables during
# hyperparameter tuning.

#------------------------------------------

## Training & Testing

# We use the createDataPartition() function
# from the caret package to create our
# training and testing sets using an
# 85/15 split.
set.seed(831) # initialize the random seed

# Generate the list of observations for the
# train dataframe
sub <- createDataPartition(y = OJ_dum$Purchase, 
                           p = 0.85, 
                           list = FALSE)

# Create our train and test sets
train <- OJ_dum[sub, ] 
test <- OJ_dum[-sub, ]

#------------------------------------------

## Analysis

# Since there is no true default model, we 
# go straight to hyperparameter tuning to
# find the optimal number of hidden nodes
# and weight decay.


### Hyperparameter Tuning Model 
### (train() in caret package)

# We can use the train() function from the 
# caret package to tune our hyperparameters. 
# Here, we will use the nnet package 
# (method = "nnet"). We can tune the size 
# and decay hyperparameters.

# Size: number of nodes in the hidden layer. 
# (Note: There can only be one hidden layer 
# using nnet)
# Decay: weight decay. to avoid overfitting, 
# adds a penalty for complexity (error + wd * SSW).
# Values typically range between 0.01 - 0.1.
# Note: nnet() does not use gradient descent,
# so there is no learning rate.

# We will use a grid search and 5-fold cross
# validation repeated 3 times.

# First, we set up the grid using the 
# expand.grid() function for the size and
# decay hyperparameters
grids <-  expand.grid(size = seq(from = 3, 
                                 to = 9, 
                                 by = 2),
                      decay = seq(from = 0,
                                  to = 0.1,
                                  by = 0.01))

grids

# Next, we set up our control object for
# input in the train() function for the
# trControl argument
ctrl <- trainControl(method = "repeatedcv",
                     number = 5, # 5 folds
                     repeats = 3, # 3 repeats
                     search = "grid") # grid search

# Next, we initialize a random seed for 
# our cross validation
set.seed(831)

# Then, we use the train() function to
# train the ANN model using 5-Fold Cross 
# Validation (repeated 3 times) to search
# over the hyperparameter grid (grids).
# We use the preProcess argument for
# range normalization.
annMod <- train(form = Purchase ~., # use all other variables to predict Purchase
                data = train[,-(13:14)], # training data
                preProcess = "range", # apply min-max normalization
                method = "nnet", # use nnet()
                trControl = ctrl, 
                maxit = 200, # increase # of iterations from default (100)
                tuneGrid = grids, # search over the created grid
                trace = FALSE) # suppress output


# We can view the Accuracy and Kappa
# across our hyperparameter grid and
# obtain the optimal values of size
# and decay
annMod

# We can visualize the tuned ANN model
# using the plotnet() function from the
# NeuralNetTools package. 
plotnet(mod_in = annMod$finalModel, # nnet object
        pos_col = "darkgreen", # positive weights are shown in green
        neg_col = "darkred", # negative weights are shown in red
        circle_cex = 4, # reduce circle size (default is 5)
        cex_val = 0.6) # reduce text label size (default is 1)


## Training Performance
# We use the predict() function to obtain
# class predictions for the Purchase
# variable using the ANN model.
tune.tr.preds <- predict(object = annMod, # tuned model
                         newdata = train) # training data

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# training dataset (train).
tune_tr_conf <- confusionMatrix(data = tune.tr.preds, # predictions
                                reference = train$Purchase, # actual
                                positive = "CH",
                                mode = "everything")

## Testing Performance
# We use the predict() function to generate 
# class predictions for our testing data set
# and evaluate model performance.
tune.te.preds <- predict(object = annMod, # tuned model
                         newdata = test) # testing data

# Next, we get performance measures using
# the confusionMatrix() function
tune_te_conf <- confusionMatrix(data = tune.te.preds, # predictions
                                reference = test$Purchase, # actual
                                positive = "CH",
                                mode = "everything")
tune_te_conf

## Goodness of Fit
# To assess if the model is balanced,
# underfitting or overfitting, we compare
# the performance on the training and
# testing. We can use the cbind() function
# to compare side-by-side.

# Overall
cbind(Training = tune_tr_conf$overall,
      Testing = tune_te_conf$overall)

# Class-Level
cbind(Training = tune_tr_conf$byClass,
      Testing = tune_te_conf$byClass)


