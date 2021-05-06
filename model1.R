#install.packages("h2o")
#install.packages("rtools40")
#install.Rtools()
#R.version.string
#writeLines('PATH="${RTOOLS40_HOME}\\usr\\bin;${PATH}"', con = "~/.Renviron")
#Sys.which("make")
#Sys.getenv("PATH")
#install.packages("jsonlite", type = "source")
set.seed(1234)

install.packages("mltools")
library(mltools)

install.packages("C:/Users/user/Documents/h2o-3.32.1.1/R/h2o_3.32.1.1.tar.gz",
                 repos = NULL, type = "source")
library(h2o)
#demo(h20.glm)
install.packages("data.table")
#library("h2o")
#library("Rtools")
install.packages('bit64', repos = "https://cran.rstudio.com")
data.table::update.dev.pkg()
library(bit64)
library(data.table)


train_dat <- read.table(file= "https://www.dropbox.com/s/bawlkeolef1bse2/train_dat.csv?dl=1", 
                        row.names = NULL, sep= ",", header= T)
test_dat <- read.table(file= "https://www.dropbox.com/s/rbjatpuk5x7dios/test_dat.csv?dl=1", 
                       row.names = NULL, sep= ",", header= T)

TT <- NROW(train_dat)
samplee <- sample(1:TT, size= 0.8*TT, replace= F)
trainn <- (train_dat[samplee, ])
y_train <- train_dat[samplee, 30]
validd <- (train_dat[-samplee, ])
y_valid <- train_dat[-samplee, 30]
test_dat <- (test_dat)

#Feature engineering
yb<-c("yb1","yb2","yb3","yb4","yb5","yb6","yb7","yb8","yb9","yb10")
colnames(trainn1)[35:44] <- c("yb1","yb2","yb3","yb4","yb5","yb6","yb7","yb8","yb9","yb10")
colnames(validd1)[33:42] <- yb
colnames(test_dat1)[33:42] <- yb


write.csv(trainn1, "trainn2.csv",row.names = FALSE, col.names  = FALSE)
write.csv(validd1, "validd2.csv",row.names = FALSE, col.names  = FALSE)
write.csv(test_dat1, "test_dat2.csv",row.names = FALSE, col.names  = FALSE)
write.csv(test_dat1, "fina_sub_benchmark.csv",row.names = FALSE, col.names  = FALSE)

?setdiff
colnames(validd1)[1] <- c("y_train")
set0 <- setdiff(colnames(trainn1), colnames(validd1))
validd1[set0] <- 0

set1 <- setdiff(colnames(validd1), colnames(trainn1))
trainn1[set1] <- 0

setdiff(colnames(trainn1), colnames(test_dat1))
set2 <- setdiff(colnames(trainn1), colnames(test_dat1))[2:5]
test_dat1[set2] <- 0


set3<-setdiff(colnames(test_dat1),colnames(trainn1))
trainn1[set3] <- 0


set4<- setdiff(colnames(validd1), colnames(test_dat1))

set5 <- setdiff(colnames(test_dat1),colnames(validd1))


#Dependent values
#y_train - we do not log transform the y variable because we assume the test y variable is not log transformed

#Numerical values- correlation - outliers(remove) 
#We use ordinal variables for those that have lots of 0s and some big values
###############################################Numerical-train
nums <- unlist(lapply(validd, is.numeric))  
numerics <-validd[,nums]
numerics1 <- numerics[,c(1,6,7,8,9,10,19,20,21,22,23,24,25,27)]
#numerics1 <- as.data.table(numerics1)
#cor(numerics1)>0.6 #X1stFlrSF - TotalBsmtSF, # y_train - TotalBsmtSF , GrLivArea, GarageArea
#colnames(numerics1)[abs(skewness(numerics1))>0.5]
#install.packages('psych')
#library('psych')
#outlier - all values have significant meaning yet, we do not know whether we need to remove them. We decide not to.

index <- which(abs(skew(numerics1))>0.6)
colnames(numerics1)[index]
colnames(numerics)
plot(numerics1$LotArea)
plot(numerics1$X1stFlrSF)
plot(numerics1$X2ndFlrSF)#
plot(numerics1$LowQualFinSF)#
plot(numerics1$GrLivArea)
plot(numerics1$GarageArea)
plot(numerics1$WoodDeckSF)#
plot(numerics1$OpenPorchSF)#
plot(numerics1$EnclosedPorch)#
plot(numerics1$X3SsnPorch)#
plot(numerics$ScreenPorch)#
plot(numerics1$MiscVal)#

#Garage area per car
numerics1$GftperC <- numerics1$GarageArea/numerics1$GarageCars
#numerics1$GftperC <- numerics1$GarageArea/numerics1$GarageCars
numerics1$GftperC[is.na(numerics1$GftperC)]<-0
#min_max
# To get a vector, use apply instead of lapply

for (i in 1:ncol(numerics1)){
  x <- numerics1[,i]
  numerics1[,i]<-(x- min(x)) /(max(x)-min(x))
}

vals <- c("X2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch",
          "X3SsnPorch","ScreenPorch","MiscVal","GftperC","PoolArea")

validd[colnames(numerics1)] <- numerics1


####################################Categorical-train
#install.packages("caret")
#install.packages("gtools")
library(caret)
library(gtools)
factors <- colnames(numerics[,-c(1,6,7,8,9,10,19,20,21,22,23,24,25,27,30,31)])
factors <- factors[-13]
#summary(train_dat$YearBuilt)
#unique(levels(factor_data[,3]))
factor_data <- trainn[,(factors)]

#Year in 10 categories just built and RemodAdd

factor_data[,3] <- cut(factor_data[,3], 10, lavels = c(1,2,3,4,5))
factor_data[,4] <- cut(factor_data[,4], 10, lavels = c(1,2,3,4,5))
factor_data[factors] <- lapply(factor_data[factors] , factor)
dmy <- dummyVars(" ~ .", data = factor_data)
factor_data1  <- data.frame(predict(dmy, newdata = factor_data))

binary <- trainn[vals]
for (i in vals){
  x<-trainn[,i]
  data <- which(x>0)
  x[data] <- 1
  x <- as.factor(x)
  binary[,i]<-x
}

#Cbind
trainn1 <- cbind(numerics1,factor_data1)
trainn1<-cbind(trainn1,binary)
trainn1 <- cbind(y_train,trainn1)
validd1 <- validd1[-112]
test_dat1 <- test_dat1[-118]
test_dat1[118]
colnames(validd1)
new <- read.csv("trainn0.csv")

###########################################################

###########################################Numerical-valid
nums <- unlist(lapply(trainn, is.numeric))  
numerics <-trainn[,nums]
numerics1 <- numerics[,c(1,6,7,8,9,10,19,20,21,22,23,24,25,27)]
#numerics1 <- as.data.table(numerics1)
#cor(numerics1)>0.6 #X1stFlrSF - TotalBsmtSF, # y_train - TotalBsmtSF , GrLivArea, GarageArea
#colnames(numerics1)[abs(skewness(numerics1))>0.5]
#install.packages('psych')
#library('psych')
#outlier - all values have significant meaning yet, we do not know whether we need to remove them. We decide not to.

index <- which(abs(skew(numerics1))>0.6)
colnames(numerics1)[index]
colnames(numerics)
plot(numerics1$LotArea)
plot(numerics1$X1stFlrSF)
plot(numerics1$X2ndFlrSF)#
plot(numerics1$LowQualFinSF)#
plot(numerics1$GrLivArea)
plot(numerics1$GarageArea)
plot(numerics1$WoodDeckSF)#
plot(numerics1$OpenPorchSF)#
plot(numerics1$EnclosedPorch)#
plot(numerics1$X3SsnPorch)#
plot(numerics$ScreenPorch)#
plot(numerics1$MiscVal)#

#Garage area per car
numerics1$GftperC <- numerics1$GarageArea/numerics1$GarageCars
#numerics1$GftperC <- numerics1$GarageArea/numerics1$GarageCars
numerics1$GftperC[is.na(numerics1$GftperC)]<-0
#min_max
# To get a vector, use apply instead of lapply

for (i in 1:ncol(numerics1)){
  x <- numerics1[,i]
  numerics1[,i]<-(x- min(x)) /(max(x)-min(x))
}

vals <- c("X2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch",
          "X3SsnPorch","ScreenPorch","MiscVal","PoolArea")

trainn[colnames(numerics1)] <- numerics1
colnames(test_dat1)[37]

####################################Categorical-train
#install.packages("caret")
#install.packages("gtools")
library(caret)
library(gtools)
factors <- colnames(numerics[,-c(1,6,7,8,9,10,19,20,21,22,23,24,25,27,30,31)])
factors <- factors[-13]
#summary(train_dat$YearBuilt)
#unique(levels(factor_data[,3]))
factor_data <- validd[,(factors)]

#Year in 10 categories just built and RemodAdd
factor_data[,3] <- cut(factor_data[,3], 10, lavels = c(1,2,3,4,5))
factor_data[,4] <- cut(factor_data[,4], 10, lavels = c(1,2,3,4,5))
factor_data[factors] <- lapply(factor_data[factors] , factor)
dmy <- dummyVars(" ~ .", data = factor_data)
factor_data1  <- data.frame(predict(dmy, newdata = factor_data))

binary <- validd[vals]
for (i in vals){
  x<-validd[,i]
  data <- which(x>0)
  x[data] <- 1
  x <- as.factor(x)
  binary[,i]<-x
}

#Cbind
validd1 <- cbind(numerics1,factor_data1)
validd1<-cbind(validd1,binary)
validd1 <- cbind(y_valid,validd1)

##################################################################test
nums <- unlist(lapply(test_dat, is.numeric))  
numerics <-test_dat[,nums]
numerics1 <- numerics[,c(1,6,7,8,9,10,19,20,21,22,23,24,25,27)]
#numerics1 <- as.data.table(numerics1)
#cor(numerics1)>0.6 #X1stFlrSF - TotalBsmtSF, # y_train - TotalBsmtSF , GrLivArea, GarageArea
#colnames(numerics1)[abs(skewness(numerics1))>0.5]
#install.packages('psych')
#library('psych')
#outlier - all values have significant meaning yet, we do not know whether we need to remove them. We decide not to.

index <- which(abs(skew(numerics1))>0.6)
colnames(numerics1)[index]
colnames(numerics)
plot(numerics1$LotArea)
plot(numerics1$X1stFlrSF)
plot(numerics1$X2ndFlrSF)#
plot(numerics1$LowQualFinSF)#
plot(numerics1$GrLivArea)
plot(numerics1$GarageArea)
plot(numerics1$WoodDeckSF)#
plot(numerics1$OpenPorchSF)#
plot(numerics1$EnclosedPorch)#
plot(numerics1$X3SsnPorch)#
plot(numerics$ScreenPorch)#
plot(numerics1$MiscVal)#

#Garage area per car
numerics1$GftperC <- numerics1$GarageArea/numerics1$GarageCars
#numerics1$GftperC <- numerics1$GarageArea/numerics1$GarageCars
numerics1$GftperC[is.na(numerics1$GftperC)]<-0
#min_max
# To get a vector, use apply instead of lapply

for (i in 1:ncol(numerics1)){
  x <- numerics1[,i]
  numerics1[,i]<-(x- min(x)) /(max(x)-min(x))
}

vals <- c("X2ndFlrSF","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch",
          "X3SsnPorch","ScreenPorch","MiscVal","GftperC","PoolArea")

test_dat[colnames(numerics1)] <- numerics1


####################################Categorical-train
#install.packages("caret")
#install.packages("gtools")
library(caret)
library(gtools)
factors <- colnames(numerics[,-c(1,6,7,8,9,10,19,20,21,22,23,24,25,27,30,31)])
factors <- factors[-13]
#summary(train_dat$YearBuilt)
#unique(levels(factor_data[,3]))
factor_data <- test_dat[,(factors)]

#Year in 10 categories just built and RemodAdd

factor_data[,3] <- cut(factor_data[,3], 10, lavels = c(1,2,3,4,5))
factor_data[,4] <- cut(factor_data[,4], 10, lavels = c(1,2,3,4,5))
factor_data[factors] <- lapply(factor_data[factors] , factor)
dmy <- dummyVars(" ~ .", data = factor_data)
factor_data1  <- data.frame(predict(dmy, newdata = factor_data))

binary <- test_dat[vals]
for (i in vals){
  x<-test_dat[,i]
  data <- which(x>0)
  x[data] <- 1
  x <- as.factor(x)
  binary[,i]<-x
}

#Cbind
test_dat1 <- cbind(numerics1,factor_data1)
test_dat1<-cbind(test_dat1,binary)
#test_dat1 <- cbind(y_valid,test_dat1)




plot(1:length(summaryF$rmsle),summaryF$rmsle)
plot(1:length(summary1$residual_deviance), as.numeric(summary1$residual_deviance))

###########################################################################Optimization

colnames(train_dat)
respond <- "y_train"
predictors <- setdiff(names(te_dat1), respond)
index_y <- which(colnames(validd1)== "y_train")
minn <- as.matrix(trainn1[,index_y]) %>% min


h2o.init()
trainn <- trainn[,1:30]
validd <- validd[,1:30]
test_dat <- test_dat[,1:29]
train_dat1

train_h2o <- as.h2o(trainn)
valid_h2o  <- as.h2o(validd)
test_h2o <- as.h2o(test_dat)

################subset of the features
colnames(train_h2o)
library(dplyr)

colnames(trainn1) %>% vals

subset(trainn1, colnames(trainn1) %>% vals)
which(colnames(trainn1) %>% vals)
which(upper.tri( cor(numerics1))>0.6)
trainn1[,-2]
trainn2 <- lapply(trainn1, as.numeric)
cor()
?cor

#h2o.no_progress()

#Maybe something larger for hidden
value <- c("GftperC","GftperC_1" )
hidden_opt1 = lapply(1:100, function(x)sample(100,sample(6), replace=TRUE))


hyper_params <- list(activation = c("Rectifier", "Tanh","RectifierWithDropout"),
                     hidden = hidden_opt1, rho = c(0.99,0.9,0.95),
                     input_dropout_ratio = c(0,0.05,0.1))

search_criteria = list(strategy = "RandomDiscrete", stopping_metric = "rmsle", 
                       stopping_tolerance = 0.00005, stopping_rounds = 10, seed = 1410,
                       max_models = 100)
dl_random_grid7 <- h2o.grid(algorithm = "deeplearning", grid_id = "idgrid1",
                           training_frame = train_h2o, validation_frame = valid_h2o,
                           x = colnames(test_dat), y = respond, epochs = 200, stopping_metric = "RMSLE",
                           stopping_tolerance = 0.00005, stopping_rounds = 5,
                           hyper_params = hyper_params, search_criteria = search_criteria)

grid1 <- h2o.getGrid(grid_id = n, "rmsle",decreasing = FALSE)
summary2 <- grid1@summary_table


best_model13@model$training_metrics

?grep
colnames(train_h2o) <- gsub("\\.","_",colnames(train_h2o))
colnames(valid_h2o) <- gsub("\\.","_",colnames(valid_h2o))
colnames(test_h2o) <- gsub("\\.","_",colnames(test_h2o))
summary(dl_random_grid7, show_stack_traces = TRUE)
n

h2o.grid()

is.numeric(trainn)
?h2o.getGrid

dl_random_grid4@summary_table$residual_deviance
n <- dl_random_grid1@grid_id

?h2o.performance()
h2o.rmsle()
summary1<-dl_random_grid@summary_table
h2o.getGrid("dl_grid_random3")
plot((dl_random_grid2@summary_table$residual_deviance) * 100000)

log(sqrt(log(sqrt(log(sqrt(as.numeric(dl_random_grid2@summary_table$residual_deviance)))))))

grid@summary_table[1,]
best_model16 <- h2o.getModel(grid@model_ids[[1]])


best_model9
print(best_model@allparameters)
print(h2o.performance(best_model, valid = T))
print(h2o.rmsle(best_model, valid = T))
m13@model$scoring_history
summaryF <- grid@summary_table
write.csv(summaryF, file = "summaryF.csv", col.names = F, row.names = F)

#############################################################
#All data

h2o.performance(dl_random_grid1@model_ids[[1]])
h2o.rmsle(dl_random_grid2)
train_dat1 <-as.h2o(train_dat)
m9 <- h2o.deeplearning(
  model_id = 'model9',
  training_frame = train_dat1,
  x = predictors,
  y = respond,
  activation = "Rectifier",
  hidden = c(40,12,10,60,59,8),
  epoch = 500,
  variable_importance = F,
  stopping_metric = "RMSLE",
  stopping_tolerance = 0.00005, stopping_rounds = 10,
  input_dropout_ratio = 0.1, categorical_encoding = "auto",
)

m13 <- h2o.deeplearning(
  model_id = 'model13',
  training_frame = train_dat1,
  x = predictors,
  y = respond,
  activation = "RectifierwithDropout",
  input_dropout_ratio = 0.05,
  hidden_dropout_ratios = 0.5,
  hidden = c(32),
  epoch = 500,
  variable_importance = F,
  stopping_metric = "RMSLE",
  stopping_tolerance = 0.00005, stopping_rounds = 10,
  categorical_encoding = "auto",
)

#############################################################
#prediction
summary(best_model9) #0.091134 - 0.125047
summary(best_model13) #0.1365 - 0.13172

#pred_dl <- ifelse(pred_dl < 0, minn, pred_dl)

#pred10 <- h2o.predict(best_model10, newdata = test_h2o)

pred13 <- h2o.predict(m13, newdata = test_h2o)
pred13 <- as.data.frame(ifelse(pred13 < 0, minn, pred13))

pred9 <- h2o.predict(m9, newdata = test_h2o)
pred9 <- as.data.frame(ifelse(pred9 < 0, minn, pred9))

surname1 <- "KIM" # first team member surname
surname2 <- "TEGUCCI" # second team member surname
file_name <- paste0(surname1, "_", surname2,".csv")
#getwd()
write.table(pred13, file= "pred_13.csv", row.names = FALSE, col.names  = FALSE)


###############################################
m1 <- h2o.deeplearning(
  model_id = 'model1',
  training_frame = validd,
  validation_frame = validd,
  x = predictors,
  y = respond,
  activation = "Rectifier",
  hidden = c(20,20),
  epoch = 50,
  variable_importance = T
)

summary(m1)

m2 <- h2o.deeplearning(
  model_id = "model2",
  training_frame = validd,
  validation_frame = validd,
  x = predictors,
  y = respond,
  hidden = c(20,20,20),
  epochs = 100,
  score_validation_samples = 100,
  stopping_rounds = 3,
  stopping_metric = "MSE",
  stopping_tolerance = 0.01
  
)


