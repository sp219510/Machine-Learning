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

setwd("~/Downloads")

# 0b. (.5) Load the caret, cluster, fpc
# and factoextra packages for use in your
# R session.
library(ggplot2) 
library(caret) 
library(cluster) # clustering
library(factoextra) # cluster validation, plots
library(fpc) # cluster validation



## 0c. (.5) Import the Bank.csv data as a dataframe
# named bank. 
bank <- read.csv(file = "Bank.csv")


## 0d. (.5) Load the Clus_Plot_Fns.RData file.
# This file contains the sil_plot() function
# to create a silhouette plot and the wss_plot()
# function to create a WSS plot.

load("Clus_Plot_Fns.RData")
#------------------------------------------
######### Solutions #########
#------------------------------------------

# 1a. (1) First, view the structure and summary
# statistic information for the bank dataframe.
str(bank)
summary(bank)


# 1b. (2) Identify any categorical variables 
# as either nominal or ordinal and then
# convert them to the appropriate type of 
# factor variable(s). 
 ## Education and Zip.Code is ordinal variable.
 ## PersonalLoan,SecuritiesAccount,CDAccount,Onlineand CreditCard are 
 ##Binary Variables.
ords <- c("Education","Family")
facs <- c("PersonalLoan", "SecuritiesAccount", "CDAccount", "Online", "CreditCard")
nums1 <- c("ID", "ZIP.Code", "Age", "Experience")
nums <- names(bank)[!names(bank) %in% c(nums1, ords, facs)]
bank[ ,facs] <- lapply(X = bank[ , facs], 
                     FUN = factor)
cen_sc <- preProcess(x = bank,
                     method = c("center", "scale"))
bank_sc <- predict(object = cen_sc,
                   newdata = bank)
cen_yj <- preProcess(x = bank,
                     method = "YeoJohnson")
bank_yj <- predict(object = cen_yj,
                   newdata = bank)

cen_bank <- preProcess(x = bank_yj,
                       method = c("center", "scale"))
bank_yjcs <- predict(object = cen_bank,
                     newdata = bank_yj)

#------------------------------------------

# 2a. (2) You will use all variables in the 
# bank dataframe EXCEPT for PersonalLoan, 
# ID and ZIP.Code in your cluster analysis. 
# Based on the data, what kind of distance 
# matrix should you use? Explain.

##We will use Grover Ditance Matrix as it can use mixed variables.



# 2b. (1) Create the distance matrix that you 
# identified in 2a.  Name the distance matrix 
# dist. Omit ID, ZIP.Code and PersonalLoan.

vars <- c(facs[2:5], nums1[3:4], ords, nums)


dist. <- daisy(x = bank[,vars],
              metric = "gower")


dist.
#------------------------------------------

# 3a. (2) Create a plot of average Silhouette
# values by k-value and obtain the
# optimal value for k. Use Ward's Method 
# HCA when creating the plot. Consider k values 
# from 2 to 15. How many clusters should you use?
# Explain. 
##As per the Silhouette value of K, we are getting max at k=2
##But since 2 will be very less custer for such big data set
## We will be using 2nd maximum in this case which will be k=9.
args(sil_plot)

sil_plot(dist_mat = dist., # distance matrix
         method = "hc", # HCA
         hc.type = "average", # average linkage
         max.k = 15) # maximum k value


# 3b. (2) In your opinion, is the k value 
# identified in 3a a good k value? 
# If not, what value of k may be more 
# appropriate and why? Explain.
## In my opinion the value of k which is identified 
##is not correct as k=2 will be very less value and
##we will not get accurate results. We will be choosing
##2nd maximum value of k which is k=8 in that way we can get
##more number of clusters and better results.
wss_plot(dist_mat = dist.,
         method = "hc",
         hc.type = "ward.D2", 
         max.k = 15) 

#------------------------------------------

# 4a. (2) Perform HCA using Ward's Method and 
# plot the resulting dendrogram.

##HCA using Wards
wards <- hclust(d = dist., 
                method = "ward.D2")
plot(wards, 
     xlab = NA, sub = NA, 
     main = "Ward's Method")
rect.hclust(tree = wards, 
            k = 8, 
            border = hcl.colors(8))
wards_clusters <- cutree(tree = wards, 
                         k = 8)

# 4b. (2) Based on the dendrogram, is there an
# obvious value for k? If so, does it
# coincide with the k you identified in
# 3a? What k do you believe you should
# use? Explain.

## According to Dendogram we can see that there is unequal distribution 
##from the distribution among the clustures we can say that there it can 
##be increased to a total of k=8 clusture size for better spacing between the 
##clustures.


# 4c. (2) Based on your answer in 4b, add 
# boxes to the plot identifying the cluster 
# assignments. Then, create a vector of 
# cluster assignments based on the chosen 
# k value. 

# Obtain average variable values for each cluster
# for (original) numeric variables
cor(x = dist., y = cophenetic(x = wards))

avg_clusters <- cutree(tree = wards, 
                       k = 8)
#------------------------------------------

# 5a. (2) Obtain cluster centroid information 
# for your Ward's HCA cluster solution for 
# all variables except ZIP.Code and ID.
vars_2 <- c(vars, facs[1])

set.seed(321)

kmeans1 <- kmeans(x = bank_yjcs[ ,vars_2], # data
                  centers = 8, # # of clusters
                  trace = FALSE, 
                  nstart = 30)

stat_HCA <- cluster.stats(d = dist., 
                           clustering = avg_clusters)

kmeans1

kmeans1$size
# 5b. (2) Using the cluster centroid information
# in 5a, describe Cluster 1 in words.
## In clusture 1


matplot(t(kmeans1$centers), 
        type = "l", 
        ylab = "", 
        xlim = c(0, 7), 
        xaxt = "n", 
        col = 1:8, 
        lty = 1:8, 
        main = "Cluster Centers")


legend("left", # left position
       legend = 1:8, # 4 lines, k = 4
       col = 1:8, # 4 colors, k = 4
       lty = 1:8, # 4 line types, k = 4
       cex = 0.6) # reduce text size

# 5c. (2) The PersonalLoan variable was omitted
# from the cluster analysis and can be used for 
# external validation. Based on the Adjusted
# Rand Index value, does the clustering
# solution do a good job separating observations
# by whether or not they have a personal loan?
# Explain.


com <- hclust(d = dist., 
              method = "complete")

com_clusters <- cutree(tree = com, k = 8)
com_clusters

table(Sales = bank$PersonalLoan, 
      Clusters = com_clusters)

cluster.stats(d = dist.,
              clustering = com_clusters,
              alt.clustering = as.numeric(bank$PersonalLoan))$corrected.rand


#------------------------------------------

# 6a. (1) Perform k-Medoids (PAM) Cluster 
# Analysis. Use the k value chosen in 4b. 
# Initialize 321 as your random seed.
wss_plot(dist_mat = dist.,
         method = "pam", 
         max.k = 15, 
         seed_no = 321)

set.seed(321)
pam_1 <- pam(x = dist., 
            k = 8, 
            diss = TRUE)
pam_1

# 6b. (2) View the Medoids. Describe Cluster 2
# in words.

bank[pam_1$medoids, ]

stat_PAM <- cluster.stats(d = dist., 
                           clustering = pam_1$clustering)

stat_PAM
## Experience is having the lest number of observation in their clusture. 
##The 2nd clusture is is having the 2nd lowest observation in it.
#------------------------------------------

# 7. (3) Compare the two clustering solutions 
# using (HCA and PAM) based on the Dunn Index,
# the average distance between clusters and
# the average distance within clusters. Based
# on these measures, which clustering solution
# is preferred? Explain.


c_stat <- c("max_diameter", "min_separation", 
             "average_inbetween", "average_within",
             "dunn")

cbind(HCA = stat_HCA[names(stat_HCA) %in% c_stat],
      PAM = stat_PAM[names(stat_PAM) %in% c_stat])

cbind(HCA = stat_HCA,
      PAM = stat_PAM)["dunn",]
##Based on the dunn Values HCA gives us better results. Beacuse 
## The dunn value of HCA is more close to 1. 

#------------------------------------------
save.image("HW2_Group12.Rdata")

