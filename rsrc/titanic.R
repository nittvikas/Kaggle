## In this script we will solve the titanic problem

## Problem Defination :  using data find out which passenger is likely to serviev

## Loading libraries required ##
# so we are solving this problem with three methods of classification
# 1. Logistic regression 
# 2. Linear Descrimental Analysis
# 3. Quadratic descrimental analysis 

library(MASS)
library(class)
library(caret)
library(tidyverse)
library(fields)

## now we will load the Data files
train_data = read_csv("data/train.csv")
head(train_data)
test_data = read_csv("data/test.csv")
head(test_data)
dim(test_data)

train_data$Survived = factor(train_data$Survived)
attach(train_data)
train_data

##next step is to select feature : Feature selection is the most important part
##of your Analysis
table(Survived, Pclass)
table(Survived, PassengerId) 
table(Survived, Age)
table(Survived, SibSp)

## Graphics is the most useful way to do this ..
# We will use the conditional box plot
bplot.xy(Survived, Age)

##comparing survival and Pclass
bplot.xy(Survived, Pclass)



## comparing survival rate and fair
bplot.xy(Survived, Fare)

# so as we decided to choose the Pclass, Age and Fair to predict servival
#First we will solve by logistic regression

log_reg = glm(formula = Survived ~ Pclass+Age+Fare, family = binomial, data = train_data)
summary(log_reg)

#predicting the values 
test_data_mod = test_data[c('PassengerId', 'Pclass', 'Age', 'Fare')]
summary(test_data_mod)

## removing the null values
test_data_mod$Age = ifelse(is.na(test_data_mod$Age), mean(test_data_mod$Age, na.rm=TRUE), test_data_mod$Age)
test_data_mod$Fare = ifelse(is.na(test_data_mod$Fare), mean(test_data_mod$Age, na.rm = TRUE), test_data_mod$Fare)


log_predic = predict(log_reg, newdata = test_data_mod, type = "response")
glm_pred = rep(0, 418)
glm_pred[log_predic > .5] = 1
test_data_mod$Survived = glm_pred

#writing result in the Kaggle format
submission = test_data_mod[, c("PassengerId", "Survived")]

head(test_data_mod)
write_csv(submission,"data/submission_logistic.csv" )


#now we will solve the same problem with the LDA
#to solve this problme we will check the model accurary by spliting the training data into 
# 80:20

# first we will clearn the data 
names(train_data)

# we are working with PassengerId, Survived, Pclass, And Fare column
train_new = train_data[,c("PassengerId", "Survived", "Pclass", "Fare", "Age")]
summary(train_new)
bplot(train_new$Age)

#first we needd to remove the null values form the Age column
# We will replace those values with mean 
train_new$Age = ifelse(is.na(train_new$Age), mean(train_new$Age, na.rm = TRUE), train_new$Age)
summary(train_new)

#now next thing we will split the train data so we can validate our mode
train = (train_new$PassengerId < 800)
train_new_train = train_new[train,]
train_new_test = train_new[!train, c("PassengerId", "Pclass", "Fare", "Age")]
dim(train_new_test)
train_new_test
train_new_test_val = train_new[!train, c("Survived")]
dim(train_new_test_val)


#as our data is ready now we will pass it to the model

# we will evaluate both model 
#LDA and QDA

lda_fit = lda(Survived~Pclass+Fare+Age, data = train_new_train)
lda_fit

#now prediction
lda_pred = predict(lda_fit, train_new_test)
lda_pre_class = lda_pred$class
test_data_lda = test_data[,c("PassengerId", "Pclass", "Fare", "Age")]
summary(test_data_lda)
test_data_lda$Fare = ifelse(is.na(test_data_lda$Fare), mean(test_data_lda$Fare, na.rm = TRUE), test_data_lda$Fare)
test_data_lda$Age = ifelse(is.na(test_data_lda$Age), mean(test_data_lda$Age, na.rm = TRUE), test_data_lda$Age)
table(lda_pre_class, train_new_test_val$Survived)
mean(lda_pre_class == train_new_test_val$Survived)
# so our model is 71.73% accurate
# now let's try to train it on the entire file and now this time we will pass it the actual test file
summary(train_new)
lda_fit = lda(Survived~Pclass+Fare+Age, data = train_new)

lda_pred = predict(lda_fit, test_data_lda)
lda_pred_class = lda_pred$class
test_data_lda$Survived = lda_pred_class
submission = test_data_lda[, c("PassengerId", "Survived")]
dim(submission)
dim(test_data)

head(submission)
write_csv(submission, "data/submission_LDA.csv")












































