#Load the dataset
#install.packages("modeldata")
library(modeldata)
Anthro <- read.csv(file = "Anthro.csv",
               stringsAsFactors =FALSE)
#isolate X and Y
#install.packages("dplyr")
library(dplyr)
y<-as.numeric(Anthro$FIFTH_ORDER)
x<-Anthro %>% select(-FIFTH_ORDER)
str(x)

#Transform factor into dummy variable
install.packages("fastDummies")
library(fastDummies)
x<- dummy_cols(x,
               remove_first_dummy = TRUE)
x<-x%>% select(-FIRST_MARKDOWN_CATEGORY)
x<-x%>% select(-SECOND_MARKDOWN_CATEGORY)
x<-x%>% select(-THIRD_MARKDOWN_CATEGORY)
x<-x%>% select(-FOURTH_MARKDOWN_CATEGORY)
x<-x%>% select(-SECOND_ORDER_DATE)
x<-x%>% select(-THIRD_ORDER_DATE)
x<-x%>% select(-FOURTH_ORDER_DATE)

#Setting the parameters
params<- list(set.seed=2110,
              eval_metric="auc",
              objective="binary:logistic")
#running xgboost
#install.packages("xgboost")
library(xgboost)
model<-xgboost(data = as.matrix(x),
               label = y,
               params = params,
               nrounds = 50,
               verbose = 1,)
#shap values
xgb.plot.shap(data = as.matrix(x),
              model=model,
              top_n = 5)
xgb.importance(model = model)


