## load the required libraries and set the seed
library(caret)
library(randomForest)
library(DataExplorer)
library(dplyr)
library(corrplot)
library(rpart)
set.seed(2018)

## The data for this project come from this source: 
## http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.
train = read.csv("C:/Users/User/Documents/coursera/pml-training.csv")
testcases = read.csv("C:/Users/User/Documents/coursera/pml-testing.csv")

## Eyeball classe variable
plot(train$classe,ylab="Frequency",
     xlab="classe levels", 
     main="Classe levels Frequency",
     col=c(12,13,14,15,16))
str(train)
dim(train)

## large dataset

## Split the train data
inTrain  <- createDataPartition(train$classe, p=0.7, list=FALSE)
train <- train[inTrain, ]
test  <- train[-inTrain, ]

## cleaining the data, remove columns with >95% NA
x <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[, x==FALSE]
## x <- sapply(test, function(x) mean(is.na(x))) > 0.95
test  <- test[, x==FALSE]
## x <- sapply(testcases, function(x) mean(is.na(x))) > 0.95
testcases  <- testcases[, x==FALSE]

## remove near zero variation
x  <-  nearZeroVar(train)
if(length(x) > 0) train <- train[, -x]
## x  <-  nearZeroVar(test)
if(length(x) > 0) test <- test[, -x]
## x  <-  nearZeroVar(testcases)
if(length(x) > 0) testcases <- testcases[, -x]

## summarise the data
## create_report(train)

## remove non-numeric columns and the first column
train <-train[, -(1:5)]
test <-test[, -(1:5)]
testcases <-testcases[, -(1:5)]

## correlation analysis, note col 54 is out Y called classe
corMatrix <- cor(train[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
corrplot(corMatrix, order = "alphabet" , type = "upper",tl.cex = 0.5)

## There are some higher correlation coefficients, causing colinearity
## Preprocessing with Principle Component Analysis
## without classe in col.54, accounting for 95% of variance
preProc <- preProcess(train[,1:53],method="pca",thresh=.95)

## 
trainingPC <- predict(preProc,train[,1:53])

FitRF <- randomForest(train$classe ~ .,   data=trainingPC, do.trace=F)
confusionMatrix(train$classe,predict(FitRF,trainingPC))

# weight per predictor
importance(FitRF)

## check with test set
testingPC <- predict(preProc,test[,1:53])
confusionMatrix(test$classe,predict(FitRF,testingPC))

## Predict classes of 20 test data
testdataPC <- predict(preProc,testcases[,1:53])
testcases$classe <- predict(FitRF,testdataPC)

## reveal prediction
testcases$classe