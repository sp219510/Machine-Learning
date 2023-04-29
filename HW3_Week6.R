#------------------------------------------
#------------------------------------------
############## Homework # 3 ##############
#------------------------------------------
#------------------------------------------

# Directions: In this assignment, you will use the 
# the Bank data file to perform Decision Tree
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

# The Bank.csv data contains records for 5000
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

## Clear your workspace
# If your workspace is NOT empty, run:
rm(list=ls())
# to clear it## Preliminary

## 0a. Set your working directory to the location
# of the Bank.csv file.
setwd("C:/Users/RAJIV KUMAR/Desktop/")

## 0b. Load the caret and rpart packages.
install.packages(c("rpart", # basic decision tree
                   "rpart.plot")) # decision tree plotting

library(caret) # Load caret libraries
library(rpart) # Load rpart libraries
library(rpart.plot)
library(Rcpp)

## 0c. Import the Bank.csv data as a dataframe
# named Bank.
Bank <- read.csv(file = "Bank.csv") 

#------------------------------------------

## 1a. (1) View structure and summary information
# for the Bank dataframe.
str(Bank) #view the structure
summary(Bank) #summary will give us the statistic information


# 1b. (2) Convert categorical variables to the 
# appropriate type of factor variable(s). 
facs <- c("Zipcode", "ID","SecuritiesAccount", "CDAccount", "Online", "CreditCard","PersonalLoan") #Creating nominal data


ords <- c("Education") #Creating ordinal data
Bank$Education <- factor(x = Bank$Education,
                         ordered = TRUE)
nums <- names(Bank)[!names(Bank) %in% c(facs, ords, "PersonalLoan", "ID", "ZIP.Code")] #Creating numerical data



#------------------------------------------

# 2a. (1) Based on the data description, what 
# will the target variable be in your 
# Decision Tree classification analysis? 

Bank$PersonalLoan <- factor(Bank$PersonalLoan) #Personal Loan is our target variable


# 2b. (1) You will use all variables as input to
# predict the target variable EXCEPT ID and
# ZIP.Code. Create a vector named vars that
# contains the names of the predictor variables.
vars <- c(facs[-1][-1], ords, nums) #Omitting ID and ZIP.Code

#------------------------------------------


## 3a. (1) Plot the distribution of the target
# variable using an appropriate plot type.
# Include a meaningful title and and axis
# labels.
par(mar=c(10,10,10,10))
plot(Bank$PersonalLoan,
     main = "PersonalLoan")



## 3b. (2) Based on your plot in 2b, does class 
# imbalance exist? If so, what is the
# minority class for the target variable?
# How might class imbalance impact classification
# analysis? Explain.

##Yes there is an significant imbalance in values of our target variables
##the minority class is the customer who accepted personal loan in 
##previous marketing campaingn 

#------------------------------------------

# 4. (2) Before performing Decision Tree
# classification analysis, what preprocessing
# and transformations are necessary? Perform 
# any steps that you identify.
## We need to first check for missing 
## Then we need to set the starting seed 
##after that we need to split the data using 85/15 rule
## The we will Subset the transformed data
## to create the training (train)
## and testing (test) datasets
any(is.na(Bank))



#------------------------------------------

# 5. (2) Split your data into training (named 
# train) and testing (named test) sets, 
# preserving the distribution of the target 
# variable. Use an 80/20 split ratio and use 
# 867 as your initial seed.
set.seed(867) 
sub <- createDataPartition(y = Bank$PersonalLoan, # target variable
                           p = 0.80, # % in training
                           list = FALSE)
train <- Bank[sub, ] 
test <- Bank[-sub, ] 

#------------------------------------------

# 6. You will perform hyperparameter tuning to
# find the optimal complexity parameter value 
# using 5-Fold Cross Validation repeated 5 times. 
# You will use a grid search to search over values
# from 0 to 0.2, counting by 0.005. 


# 6a. (1) Set up the grid described above
# for a grid search. 
grids <- expand.grid(cp = seq(from = 0,
                              to = 0.2,
                              by = 0.005))
grids



# 6b. (1)  Set up your 5-Fold Cross Validation 
# (repeated 5 times) grid search hyperparameter
# tuning model. 

ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 5,
                     search = "grid")

# 6c. (1) Initialize 5309 as your random seed, then
# perform the Decision Tree classification with
# hyperparameter tuning. 
# Hint: use your vars vector. 

set.seed(5309)
DTFit <- train(form = PersonalLoan ~ ., # use all variables in data to predict
               data = train[ ,c(vars, "PersonalLoan")], # include only relevant variables
               method = "rpart", # use the rpart package
               trControl = ctrl, # control object
               tuneGrid = grids) # custom grid object for search


# 6d. (2) What is the optimal complexity parameter
# value? What are the average accuracy and 
# kappa values for this cp value?

DTFit # all results
##Accuracy was used to select the optimal model using the largest value.
##The final value used for the model was cp = 0.2.

DTFit$results[DTFit$results$cp %in% DTFit$bestTune,] #best result


# 6e. (2) How does the complexity parameter value
# impact the size of the tree? Based on the optimal
# cp value identified in 6d, is the tree larger or
# smaller? Explain.

##complexity parameter imposes a penalty to the tree for having too many splits.


# 6f. (1) What are the 3 most important 
# variables in your optimal Decision Tree
# model?


#------------------------------------------

# 7a. (2) Obtain predictions and performance 
# information for the training set. Use the 
# class with the smallest number of observations
# as the positive class all available
# performance information in your output.

Bank.rpart <- rpart(formula = PersonalLoan ~ ., # Y ~ all other variables in dataframe
                  data = train[ ,c(vars, "PersonalLoan")], # include only relevant variables
                  method = "class")
Bank.rpart

Bank.trpreds <- predict(object = Bank.rpart, # DT model
                        newdata = train, # training data
                        type = "class") # class predictions

# 7b. (2) Obtain predictions and performance 
# information for the testing set. Use the 
# class with the smallest number of observations
# as the positive class all available
# performance information in your output.
Bank.tepreds <- predict(object = Bank.rpart, # DT model
                        newdata = test, # testing data
                        type = "class")

#------------------------------------------

# 8a. (2) Describe the performance of the model. 
# Is this a good model? Explain.

DTFit$overall[c("Accuracy", "Kappa")] #overall performance based on our accuracy and kappa value
DTFit$byClass #Class-Level Performance

# 8b. (2) Evaluate the goodness of fit of the model. 
# How would you describe the model fit? Explain. 

# Overall
cbind(Training = DTFit$overall,
      Testing = DTFit$overall)

# Class-Level
cbind(Training = DTFit$byClass,
      Testing = DTFit$byClass)

# Both models had comparable/consistent performance on the training and testing 
# sets, suggesting a balanced fit.



# 8c. (2) Based on the performance and fit of the
# Decision Tree classification model and the business 
# objective, would you recommend that the bank use 
# this model? Why or why not? Explain.  



#------------------------------------------