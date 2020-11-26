library(data.table)
library(dplyr)
#rm(list=ls())

setwd("~/Google Drive/Documents/Academic Research/Baidu_2019/SGD-Diagnostic/code/R_momentum/")
news.dat = fread("~/Google Drive/Documents/Academic Research/Baidu_2019/SGD-Diagnostic/data/OnlineNewsPopularity.csv")

# extract (57) predictive features
# easier just to scale everything.
X = scale(
  as.matrix(
    news.dat %>% select(-c(url, abs_title_subjectivity, abs_title_sentiment_polarity, shares))
    )
  )

# generate categorical predictive variable
Y = as.numeric(news.dat$shares)
Y[Y < 1400]  = 0
Y[Y >= 1400] = 1

# create train/test
N = nrow(X)
sample = sample.int(n=N, size=floor(.80*N), replace=FALSE)
X_train <- X[sample, ]
Y_train <- Y[sample]
X_test <- X[-sample, ]
Y_test <- Y[-sample]

# create format for iSGD
data.news.scaled <- list(model_name="binomial", 
                    X_train=X_train, Y_train=Y_train,
                    X_test=X_test, Y_test=Y_test,
                    glm_link=function(x) expit(x),
                    du_glm_link=function(x) du_expit(x))

#saveRDS(data.news.scaled, "~/Google Drive/Documents/Academic Research/Baidu_2019/SGD-Diagnostic/data/news.scaled.rds")
