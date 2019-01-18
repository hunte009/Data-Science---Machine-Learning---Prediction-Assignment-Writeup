---
title: "Prediction Assignment Writeup"
author: "Arjen Hunter"
date: "11 January 2019"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Introduction
A group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available on the website: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
It is the goal of this project to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We will use any of the other variables to predict with. In this report we describe how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we made. We will also use our prediction model to predict 20 different test cases that have been supplied in the assignment.

## Method used
The data that was made available for this project, consists of a training dataset and a testing dataset. Here is the method we have used:

1) Loading the data set;

2) Split the data;

3) Cleaning the data;

4) Look for covariance;

5) Build a model;

6) Verify the model;

7) Apply the model;

## Loading the data.
First of all we must set the seed and load all required libraries.
```{r include=FALSE}
## load the required libraries and set the seed
library(caret)
library(randomForest)
library(DataExplorer)
library(dplyr)
library(corrplot)
library(rpart)
set.seed(2018)
```

Then we load the data set and eyeball the characteristics of the data. Most of the exploratory data analysis is not shown here. It was performed using the DataExplorer package. A package that automatically scans through the data and does data profiling, reporting the results in a separate file.
```{r}
train = read.csv("C:/Users/User/Documents/coursera/pml-training.csv")
testcases = read.csv("C:/Users/User/Documents/coursera/pml-testing.csv")
## Eyeball classe variable
plot(train$classe,ylab="Frequency",
     xlab="classe levels", 
     main="Classe levels Frequency",
     col=c(12,13,14,15,16))
```

In the Classe levels plot we see that there are 5 levels to the classe label and all five levels have a substantial amount of measurement. Some of the data may be missing (NA).
```{r}
sum(sapply(train, function(x) mean(is.na(x))) > 0.95)
```

The dataset contains 67 labels with ?95% of NA's and potentially a number of labels with very little variance.
```{r echo=TRUE}
x  <-  nearZeroVar(train)
```

A total of 51 labels contain very little variance. Including those labels would increase the risk of overfitting the model, resulting in a unnecessarily large out of sample error when applying the model to predictions. Since both high level of NA and low variance will be removed from the dataset that we'll be building our model upon, the overlap between the two groups is of no interest to us. 

## Split the data
Use cross-validation method to split the training dataset into separate set for developing the model and the other to prove the developed model

    a) 70% of the original data is used for model building (training data)

    b) 30% of the data is used for testing (testing data);
```{r}
inTrain  <- createDataPartition(train$classe, p=0.7, list=FALSE)
train <- train[inTrain, ]
test  <- train[-inTrain, ]
```

## Cleaning the data

The training data contains 160 labels. We need to clean the data by 

  a) excluding labels containing mostly NA (i.e. empty cells);
  
  b) excluding variables which have little variation;
  
Note there are 3 sets of data that must be subjected to the same cleaning procedure, the training data, the testing data and the testcase data sets. The cleaning procedure is determined on the trainingset alone, whereas it is applied to all three sets.
```{r}
## cleaining the data, remove columns with >95% NA.
x <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[, x==FALSE]
test  <- test[, x==FALSE]
testcases  <- testcases[, x==FALSE]

## remove near zero variation.
x  <-  nearZeroVar(train)
if(length(x) > 0) train <- train[, -x]
if(length(x) > 0) test <- test[, -x]
if(length(x) > 0) testcases <- testcases[, -x]

## remove non-numeric columns and the first column with serial number.
train <-train[, -(1:5)]
test <-test[, -(1:5)]
testcases <-testcases[, -(1:5)]
```
We now have removed labels containing more than 95% of NA, labels having very little variation and the first 5 columns, containing labels that are not usefull in this analysis. Such as names and dates. 

## Look for covariance
By building a correlation matrix we show the level of covariance in the data. If many labels are highly correlated, they will negatively influence the model. Perform Principle Component Analysis will be used to further reduce the number of variables.
```{r}
## correlation analysis, note col 54 is out Y called classe
corMatrix <- cor(train[, -54])
corrplot(corMatrix, order = "alphabet", type = "upper", tl.cex = 0.5)
```

As can be seen in the correlation matrix, there is a number of significant correlations within the dataset, so is makes sense to perform PCA.

```{r}
preProc <- preProcess(train[,1:53],method="pca",thresh=.95)
```
The pre-processed model now stored in preProc was created from 13737 samples and 53 variables. PCA uses 25 components to capture 95 percent of the variance.

## Build a model
Build a model using the random forest method using the pre-processed training set.
```{r}
trainingPC <- predict(preProc,train[,1:53])
FitRF <- randomForest(train$classe ~ ., data=trainingPC, do.trace=F)
confusionMatrix(train$classe,predict(FitRF,trainingPC))
```
Here we see our model has achieved a very high accuracy 95% CI : (0.9997, 1). The confidence interval includes 1.
Note the complete separation in the selection matrix, there are no errors. Which is confirmed by the perfect sensitivity and specificity for all levels of classe.

##Verify the model
Verify the model with the testing data set
```{r}
testingPC <- predict(preProc,test[,1:53])
confusionMatrix(test$classe,predict(FitRF,testingPC))
```
ALso here we see our model has performed very well, achieving a very high accuracy of 95% CI : (0.9991, 1). Which is only marginally lower compared to the perfomance on the training set. Note also here the complete separation in the selection matrix.

## Apply the model
Apply the model to estimate classes of 20 observations
```{r}
testdataPC <- predict(preProc,testcases[,1:53])
testcases$classe <- predict(FitRF,testdataPC)
testcases$classe
```
Given the high level of accuracy of the model, the prediction should be very good indeed.

## Summary and conclusion
A large part of the data was removed during the cleaning process, being either NA or having very little variance. In the remaining there were a number of data labels having significant levels of correlation, which was subsequently reduced, using principal component analysis. From 160 labels, with 19622 samples, the data was reduced to 13737 samples and 53 variables PCA needed 25 components to capture 95 percent of the variance. Using the reduced dataset a model was built, that was nearly perfect, achieving scores of 100% for sensitivity and specificity. This has resulted in similarly perfect results on the test set. As a result of which the out of sample error will be near zero as well. Based on the high accuracy of the model, we have a lot of confidence in our prediction being correct.

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.