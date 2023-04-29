getwd()


library(ggplot2)
library(caret)
library(cluster) # clustering
library(factoextra) # cluster validation, plots
library(fpc) # cluster validation
library(rpart) # basic decision tree
library(rpart.plot) # decision tree plotting
library(caretEnsemble) # training multiple models, custom ensembles


# Importing the .csv file to the global Enviornment

Home_Equity <- read.csv("loan_home_equity.csv", header = TRUE, sep = "," )
Home_Equity

str(Home_Equity)

#Converting integer variable to numeric variables

Home_Equity$MDUE <- as.numeric(Home_Equity$MDUE)
Home_Equity$LOAN <- as.numeric(Home_Equity$LOAN)
Home_Equity$HVAL <- as.numeric(Home_Equity$HVAL)
Home_Equity$YOJ <- as.numeric(Home_Equity$YOJ)
Home_Equity$NDEROG <- as.numeric(Home_Equity$NDEROG)
Home_Equity$NDELINQ <- as.numeric(Home_Equity$NDELINQ)
Home_Equity$CLAGE <- as.numeric(Home_Equity$CLAGE)
Home_Equity$NCL <- as.numeric(Home_Equity$NCL)
Home_Equity$DEBTR <- as.numeric(Home_Equity$DEBTR)
Home_Equity$NINQ <- as.numeric(Home_Equity$NINQ)

#Converting integer "Defaulter" to factor
Home_Equity$DEFAULT <- as.factor(Home_Equity$DEFAULT)
Home_Equity$PURPOSE <- as.factor(Home_Equity$PURPOSE)
Home_Equity$JOB <- as.factor(Home_Equity$JOB)

str(Home_Equity)

summary(Home_Equity)

Home_Equity <- na.omit(Home_Equity)
dim(Home_Equity)


par(mfrow = c(2,3))
boxplot(Home_Equity$LOAN, main = "LOAN")
boxplot(Home_Equity$MDUE, main = "MDUE")
boxplot(Home_Equity$HVAL, main = "HVAL")
boxplot(Home_Equity$YOJ, main = "YOJ")
boxplot(Home_Equity$NDEROG, main = "NDEROG")
boxplot(Home_Equity$NDELINQ, main = "NDELINQ")
boxplot(Home_Equity$CLAGE, main = "CLAGE")
boxplot(Home_Equity$NINQ, main = "NINQ")
boxplot(Home_Equity$NCL, main = "NCL")
boxplot(Home_Equity$DEBTR, main = "DEBTR")

outliers_remover <- function(a){
  df <- a
  aa<-c()
  count<-1
  for(i in 1:ncol(df)){
    if(is.numeric(df[,i])){
      Q3 <- quantile(df[,i], 0.75, na.rm = TRUE)
      Q1 <- quantile(df[,i], 0.25, na.rm = TRUE) 
      IQR <- Q3 - Q1  #IQR(df[,i])
      upper <- Q3 + 1.5 * IQR
      lower <- Q1 - 1.5 * IQR
      for(j in 1:nrow(df)){
        if(is.na(df[j,i]) == TRUE){
          next
        }
        else if(df[j,i] > upper | df[j,i] < lower){
          aa[count]<-j
          count<-count+1                  
        }
      }
    }
  }
  df<-df[-aa,]
}

Home_Equity_Out <- outliers_remover(Home_Equity)
boxplot(Home_Equity_Out)
dim(Home_Equity_Out)


par(mfrow = c(2,2))
barplot(table(Home_Equity$PURPOSE), main = "PURPOSE")
barplot(table(Home_Equity$JOB), main = "JOB")
barplot(table(Home_Equity$DEFAULT), main = "DEFAULT")


Home_Equity <- Home_Equity[!(Home_Equity$PURPOSE ==""),]
Home_Equity <- Home_Equity[!(Home_Equity$JOB ==""),]


par(mfrow = c(1,2))
barplot(table(Home_Equity$PURPOSE), main = "PURPOSE")
barplot(table(Home_Equity$JOB), main = "JOB")
dim(Home_Equity)


cor(Home_Equity[c("LOAN", "MDUE", "HVAL", "YOJ", "NDEROG", "NDELINQ", "CLAGE", "NINQ", "NCL", "DEBTR")])

library(corrplot)

Home_Equity$NDEROG <- NULL
Home_Equity$NDELINQ <- NULL
# there is multicollinearity between MORTDUE and VALUE
Home_Equity$MDUE <- NULL # delete MORTDUE
dim(Home_Equity)


#Creating Training Home_Equity

input_ones <- Home_Equity[which(Home_Equity$DEFAULT == 1), ] #all 1's
input_zeros <- Home_Equity[which(Home_Equity$DEFAULT == 0), ] # all 0's
set.seed(100) # for repeatability of sample
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7 * nrow(input_ones)) #1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7 * nrow(input_zeros)) #0's for training
#pick as many as 0's and 1's
training_ones <- input_ones[input_ones_training_rows, ]
training_zeros <- input_zeros[input_zeros_training_rows, ]
#row bind the 1's and 0's
trainingHome_Equity <- rbind(training_ones, training_zeros)
#create test Home_Equity
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
#row bind the 1's and 0's
testHome_Equity <- rbind(test_ones, test_zeros)

table(trainingHome_Equity$DEFAULT)


prop.table(table(trainingHome_Equity$DEFAULT))

library(rpart)
treeMod <- rpart(DEFAULT ~., data = trainingHome_Equity)
pred_treeMod <- predict(treeMod, newdata = testHome_Equity)


accuracy.meas(testHome_Equity$DEFAULT, pred_treeMod[,2])

# check the accuracy using ROC curve
roc.curve(testHome_Equity$DEFAULT, pred_treeMod[,2], plotit = F)

#check table
table(trainingHome_Equity$DEFAULT)

# Under-sampling
data_balanced_under <- ovun.sample(DEFAULT ~., data = trainingHome_Equity, method = "under", N = 420)$data
table(data_balanced_under$DEFAULT)

# Over-sampling
data_balanced_over <- ovun.sample(DEFAULT ~., data = trainingHome_Equity, method = "over", N = 4288)$data
table(data_balanced_over$DEFAULT)

# Do both under-sampling and over-sampling
# In this case, the minority class is oversampled with replacement and majority class is undersampled without replacement.
data_balanced_both <- ovun.sample(DEFAULT ~., data = trainingHome_Equity, method = "both", p = 0.5, N = 2354)$data
table(data_balanced_both$DEFAULT)


# built decision tree models
tree.under <- rpart(DEFAULT ~., data = data_balanced_under)
tree.over <- rpart(DEFAULT  ~., data = data_balanced_over)
tree.both <- rpart(DEFAULT  ~., data = data_balanced_both)


# make predictions on test data
pred_tree.under <- predict(tree.under, newdata = testHome_Equity)
pred_tree.over <- predict(tree.over, newdata = testHome_Equity)
pred_tree.both <- predict(tree.both, newdata = testHome_Equity)


# AUC
par(mfrow = c(2,2))
roc.curve(testHome_Equity$DEFAULT, pred_tree.under[,2], col = "GREEN", main = "ROC curve of under")
roc.curve(testHome_Equity$DEFAULT, pred_tree.over[,2], col = "BLUE", main = "ROC curve of over")
roc.curve(testHome_Equity$DEFAULT, pred_tree.both[,2], col = "RED", main = "ROC curve of both")

summary(testHome_Equity)


write.csv(testHome_Equity, "C:\\Term4\\Business Consulting\\Week2\\Home_Equity\\HMEQ\\HomeEQUITY.csv")


