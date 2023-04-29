#------------------------------------------
#------------------------------------------
############## Homework # 2 ##############
#------------------------------------------
#------------------------------------------

# Directions: In this assignment, you will use the 
# the Bank.csv data file to perform Cluster 
# Analysis. 
# If an answer requires a written response, use '##' 
# before your answer. You will submit (1) .R 
# R Script file and (1) .RData file, which contains all
# objects in your workspace created while completing the
# homework assignment. If you have extra objects in your
# workspace, clear your workspace, run your HW file and
# then save your workspace using save.image(). 

# Note: Your files (.R and .RData) should be named using 
# the following format: HW#_Group. For instance, the 
# 3rd HW for Group 15 would be: HW3_Group15.
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
# past. The bank would like to group customers
# into clusters for their new personal loan
# marketing campaign. 

#------------------------------------------

## Preliminary

# 0a. (.5) Set your working directory.
setwd("C:/Users/Desktop/")


# 0b. (.5) Load the caret, cluster, fpc
# and factoextra packages for use in your
# R session.
install.packages("caret") #Install caret packages
install.packages("cluster") #Install cluster packages
install.packages("fpc") #Install fpc packages
install.packages("factoextra") #Install factoextra packages
library(caret) # Load caret libraries
library(cluster) # Load cluster libraries
library(factoextra) # Load factoextra libraries
library(fpc) # Load fpc libraries


## 0c. (.5) Import the Bank.csv data as a dataframe
# named bank. 
bank <- read.csv(file = "Bank.csv")

## 0d. (.5) Load the Clus_Plot_Fns.RData file.
# This file contains the sil_plot() function
# to create a silhouette plot and the wss_plot()
# function to create a WSS plot.
load(file = "Clus_Plot_Fns.RData")

#------------------------------------------
######### Solutions #########
#------------------------------------------

# 1a. (1) First, view the structure and summary
# statistic information for the bank dataframe.
str(bank) #view the structure
summary(bank) #summary will give us the statistic information


# 1b. (2) Identify any categorical variables 
# as either nominal or ordinal and then
# convert them to the appropriate type of 
# factor variable(s). 

facs <- c("Zipcode", "ID","SecuritiesAccount", "CDAccount", "Online", "CreditCard","PersonalLoan") #nominal data
ords <- c("Education")#ordinal data
nums <- names(bank)[!names(bank) %in% c(facs, ords)] #numerical data


#------------------------------------------

# 2a. (2) You will use all variables in the 
# bank dataframe EXCEPT for PersonalLoan, 
# ID and ZIP.Code in your cluster analysis. 
# Based on the data, what kind of distance 
# matrix should you use? Explain.
vars <- c(facs[1][2][6], ords, nums) #omit PersonalLoan, ID and ZIP.Code
## Since we have factors and numeric variables, we can use the daisy() function from the cluster 
## package to create a distance matrix using Gower distance


# 2b. (1) Create the distance matrix that you 
# identified in 2a.  Name the distance matrix 
# dist. Omit ID, ZIP.Code and PersonalLoan.
bank_yj <- preProcess(x = bank,
                     method = "YeoJohnson")
dist <- daisy(x = bank_yj[, vars], 
              metric = "gower")
summary(dist)

#------------------------------------------

# 3a. (2) Create a plot of average Silhouette
# values by k-value and obtain the
# optimal value for k. Use Ward's Method 
# HCA when creating the plot. Consider k values 
# from 2 to 15. How many clusters should you use?
# Explain.
args(sil_plot)
sil_plot(dist_mat = dist, # distance matrix
         method = "hc", # HCA
         hc.type = "average", # average linkage
         max.k = 15) # maximum k value

# 3b. (2) In your opinion, is the k value 
# identified in 3a a good k value? 
# If not, what value of k may be more 
# appropriate and why? Explain.



#------------------------------------------

# 4a. (2) Perform HCA using Ward's Method and 
# plot the resulting dendrogram.





# 4b. (2) Based on the dendrogram, is there an
# obvious value for k? If so, does it
# coincide with the k you identified in
# 3a? What k do you believe you should
# use? Explain.




# 4c. (2) Based on your answer in 4b, add 
# boxes to the plot identifying the cluster 
# assignments. Then, create a vector of 
# cluster assignments based on the chosen 
# k value. 



#------------------------------------------

# 5a. (2) Obtain cluster centroid information 
# for your Ward's HCA cluster solution for 
# all variables except ZIP.Code and ID.




# 5b. (2) Using the cluster centroid information
# in 5a, describe Cluster 1 in words.





# 5c. (2) The PersonalLoan variable was omitted
# from the cluster analysis and can be used for 
# external validation. Based on the Adjusted
# Rand Index value, does the clustering
# solution do a good job separating observations
# by whether or not they have a personal loan?
# Explain.




#------------------------------------------

# 6a. (1) Perform k-Medoids (PAM) Cluster 
# Analysis. Use the k value chosen in 4b. 
# Initialize 321 as your random seed.





# 6b. (2) View the Medoids. Describe Cluster 2
# in words.



#------------------------------------------

# 7. (3) Compare the two clustering solutions 
# using (HCA and PAM) based on the Dunn Index,
# the average distance between clusters and
# the average distance within clusters. Based
# on these measures, which clustering solution
# is preferred? Explain.



#------------------------------------------
