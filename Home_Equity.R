
library(readr)
library(tidyverse)
he <- read_csv("data2.csv", na = c("", "NA", "N/A"))
#he <- read_csv("loan_home equity.csv")
#------------------------------------------
## Data & Variables
#------------------------------------------

## Preview the data
head(he)
tail(he)

## Describing the data
# Structure of the R object
str(he)

# Rows
nrow(he)
# Columns
ncol(he)

summary(he)

str(he)

colSums(is.na(he))

he= na.omit(he)
he = he[,-1]

he$Default= as.factor(he$Default)

summary(he)

str(he)

#------------------------------------------
## Data Exploration
#------------------------------------------

# Obtain data structure
str(he)

## Summary statistics
summary(object = he)

# Mode
modefun <- function(x){
  if(any(tabulate(match(x, unique(x))) > 1)){
    outp <- unique(x)[which(tabulate(match(x, unique(x))) == max(tabulate(match(x, unique(x)))))]
  } else {
    outp <- "No mode exists!"}
  return(outp)
}

# apply the function to a single variable, CompPrice
modefun(x = he$Loan_Amount)

# apply to all variables in the cs dataframe,
# using the lapply() function
lapply(X = he, 
       FUN = modefun)


# Variance 
# apply the function to a single variable, Income
var(x = he$Loan_Amount)

# apply to all numeric variables in the cs dataframe,
# using the sapply() function
sapply(X = he[,-c(4,5,12)], 
       FUN = var)

# Standard deviation
sd(x = he$Loan_Amount)
# apply to all numeric variables in the cs dataframe,
# using the sapply() function
sapply(X = he[,-c(4,5,12)], 
       FUN = sd)


## Frequency Tables
# 1-Way frequency table for a single variable, 
# ShelveLoc
table(x = he$Reason_HE)
# 1-Way frequency tables for all factor
# variables using lapply()
lapply(X = he[ ,c('Reason_HE','Occupation','Default')], FUN = table)

# 2-Way Frequency Table
crosstabs <- table(he$Reason_HE,he$Default)
# crosstabs <- table(Level = cs$Sales_Lev, 
#                   Location = cs$ShelveLoc)
crosstabs

#------------------------------------------
## Data Visualization
#------------------------------------------


ggplot(he,aes(x=Default))+geom_bar(fill='red')
ggplot(he,aes(x=Reason_HE))+geom_bar(fill='red')
ggplot(he,aes(x=Occupation))+geom_bar(fill='red')

library(ROSE)
heml <- ovun.sample(Default~., data=he, method = "over")$data
ggplot(heml,aes(x=Default))+geom_bar(fill='red')


ggplot(data=heml)+ geom_bar(mapping=aes(x=Default,fill=Reason_HE), position = "dodge")
ggplot(data=heml)+ geom_bar(mapping=aes(x=Default,fill=Occupation), position = "dodge")
ggplot(data=heml)+ geom_bar(mapping=aes(x=Reason_HE,fill=Occupation), position = "dodge")

boxplot(heml[,-c(4,5,13)])

library(caret)
process <- preProcess(as.data.frame(heml), method=c("range"))

heml <- predict(process, as.data.frame(heml))

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  A <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - A)] <- NA
  y[x > (qnt[2] + A)] <- NA
  y
}

heml$Mort_Bal=remove_outliers(heml$Mort_Bal)
heml$Loan_Amount=remove_outliers(heml$Loan_Amount)
heml$Home_Val = remove_outliers(heml$Home_Val)
heml$YOJ =remove_outliers(heml$YOJ)
heml$Num_Derog =remove_outliers(heml$Num_Derog)
heml$Num_Delinq =remove_outliers(heml$Num_Delinq)
heml$CL_Age =remove_outliers(heml$CL_Age)
heml$Num_Inq =remove_outliers(heml$Num_Inq)
heml$Num_CL =remove_outliers(heml$Num_CL)
heml$Debt_Inc =remove_outliers(heml$Debt_Inc)
heml$Num_Derog =remove_outliers(heml$Num_Derog)
boxplot(heml[,-c(4,5,13)])

heml = na.omit(heml)


#------------------------------------------
## Decision tree
#------------------------------------------

# Initialize random seed
set.seed(8103) 
# Create list of training indices
sub <- createDataPartition(y = heml$Default, # target variable
                           p = 0.80, # % in training
                           list = FALSE)

# Subset the transformed data
# to create the training (train)
# and testing (test) datasets
train <- heml[sub, ] # create train dataframe
test <- heml[-sub, ] # create test dataframe

library(rpart)
library(rpart.plot)

FD.rpart <- rpart(formula = Default ~ ., # Y ~ all other variables in dataframe
                  data = train, # include only relevant variables
                  method = "class")
FD.rpart$variable.importance
prp(x = FD.rpart, # rpart object
    extra = 2) # include proportion of correct predictions
base.trpreds <- predict(object = FD.rpart, # DT model
                        newdata = train[,-13], # training data
                        type = "class") # class predictions
DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train$Default, # actual
)
DT_train_conf

base.tepreds <- predict(object = FD.rpart, # DT model
                        newdata = test[,-13], # testing data
                        type = "class")
DT_test_conf <- confusionMatrix(data = base.tepreds, # predictions
                                reference = test$Default, # actual
)
DT_test_conf

# Overall
cbind(Training = DT_train_conf$overall,
      Testing = DT_test_conf$overall)

# Class-Level
cbind(Training = DT_train_conf$byClass,
      Testing = DT_test_conf$byClass)

grids <- expand.grid(cp = seq(from = 0,
                              to = 0.05,
                              by = 0.005))
grids
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     search = "grid")
set.seed(8103)
DTFit <- train(form = Default ~ ., 
               data = train,
               method = "rpart", 
               trControl = ctrl, 
               tuneGrid = grids) 
DTFit # all results

DTFit$results[DTFit$results$cp %in% DTFit$bestTune,] #best result


# We can plot the cp value vs. Accuracy
plot(DTFit)


confusionMatrix(DTFit)
DTFit$finalModel$variable.importance

tune.trpreds <- predict(object = DTFit,
                        newdata = train[,-13],
                        type = "raw")

DT_trtune_conf <- confusionMatrix(data = tune.trpreds, # predictions
                                  reference = train$Default,
                                  mode = "everything")
DT_trtune_conf
tune.tepreds <- predict(object = DTFit,
                        newdata = test[,-13])

# We use the confusionMatrix() function
# from the caret package
DT_tetune_conf <- confusionMatrix(data = tune.tepreds, # predictions
                                  reference = test$Default,
                                  mode = "everything")
DT_tetune_conf

# Overall
cbind(Training = DT_trtune_conf$overall,
      Testing = DT_tetune_conf$overall)

# Class-Level
cbind(Training = DT_trtune_conf$byClass,
      Testing = DT_tetune_conf$byClass)

write.csv(heml,'D:/nnewhe.csv',row.names = FALSE)
