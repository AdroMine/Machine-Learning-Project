# Predicting Quality of Exercise
Enelen Brinshaw  

## Executive Summary

With fitness trackers becoming more and more ubiquitous, people are increasingly quantifying how much exercise they are doing. However, an important parameter missing from these discussions is how *well* they are doing it. This research aims to add to this dimension by using data from the fitness trackers to classify how well the exercise is being done. A random forest model was created on the model with an out of sample accuracy of 99.68%.   

## Data Overview

![Data collected by sensors](./sensors.png)   

    


The data consists of 160 variables of measurements taken by the sensors on the arm, wrist, belt and the dumbbell. The exercise was then classified in 5 groups: Class A to Class E, with Class A denoting the exercise was done well, as per the instructions, and the other classes denoting common mistakes made while exercising. More information about the data can be found [here](http://groupware.les.inf.puc-rio.br/har).   

## Loading and Pre-Processing Data



First we need to load and process our data.  


```r
original <- read.csv("pml-training.csv")
require(pander)
pander(dim(original))
```

_19622_ and _160_

There are 160 variables. Exploring the dataset reveals that many variables have mostly NA values in them. Moreover, the first five variables are just user identification and timestamp, so not of much use in building our model. Moreover, many variables were numeric, but were coerced to factors when loading, that also needs to be corrected. Many factor variables also had no values just " ","","." or "#DIV/0!", and they will all be removed as well.   


```r
k <- apply(original,2,function(x) sum(is.na(x))) # find number of NA's for all columns
df <- original[which(k==0)]  # remove columns which were mostly NAs (19216/19622 NAs)
df <- df[-(1:5)]   # remove name and timestamp
div0 <- c(9,12,15,49,52,55,71,74,77) # variables that had #DIV/0! values mostly
df <- df[-div0] # remove them

require(tidyr) # for extract_numeric function
for(i in 2:78){
     df[,i] <- extract_numeric(df[,i]) # convert factors to numeric
}

k2 <- apply(df,2,function(x) sum(is.na(x))) # some numeric again mostly na
df <- df[which(k2==0)] # so removing columns with mostly na
pander(dim(df))
```

_19622_ and _55_

That's a hundred and five variables removed now. The remaining 55 variables will now be used to train our algorithm.  

Now let us partition our data into a training set and validation set (we already have a test set), using a 60:40 split.   


```r
set.seed(1234)
require(caret)
inTrain <- createDataPartition(df$classe,p = 0.6, list = FALSE)
training <- df[inTrain,]
validation <- df[-inTrain,]
```


## Model Selection

To select the model, various models were created using random forests, random forests with boosting, logistic regression, linear discriminant analysis, and support vector machines. I will only display the accuracy measures of the different models here (building them all again for the report will take too much time), and the full details of the final model selected after that.    


| Algorithm                       |   Accuracy |
|:--------------------------------|-----------:|
| Random Forest                   |     99.54% |
| Random Forest (with resampling) |     99.76% |
| Linear Discriminant Analysis    |     71.37% |
| gbm (boosted trees)             |     98.81% |
| Ridge                           |     46.56% |
| Support Vector Machine          |     99.57% |
   
   
Looking at these results, random forests and support vector machines provide the best results. Further tweaking of these two models lead to a random forest based model providing the best results.   


```r
require(randomForest)
library(doParallel) # for faster parallel processing (Windows)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
# training Model
finalFit <- train(training[,-55],training$classe,
                  method = "parRF",
                  trControl = trainControl("cv",5),
                  tuneGrid = data.frame(mtry = c(7,14,28)))
stopCluster(cl) # parallel processing over
# predicting on validation set
pred.Final <- predict(finalFit,validation)

# creating a confusion matrix for our results
cf <- confusionMatrix(pred.Final,validation$classe)
```
  
  
The model was built using the "parRF" method which stands for parallel random forest. 5-folds cross-validation was used, and the given 3 values of mtry were to be used.   

Now let us view the model.   


```r
finalFit
```

```
## Parallel Random Forest 
## 
## 11776 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9420, 9420, 9421, 9422, 9421 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    7    0.9938866  0.9922665  0.003038950  0.003845077
##   14    0.9951602  0.9938775  0.002625955  0.003322295
##   28    0.9943958  0.9929106  0.001569812  0.001986093
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 14.
```

```r
cf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    3    0    0    0
##          B    0 1513    6    0    2
##          C    0    2 1362   10    0
##          D    0    0    0 1276    2
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9968          
##                  95% CI : (0.9953, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.996           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9967   0.9956   0.9922   0.9972
## Specificity            0.9995   0.9987   0.9981   0.9997   1.0000
## Pos Pred Value         0.9987   0.9947   0.9913   0.9984   1.0000
## Neg Pred Value         1.0000   0.9992   0.9991   0.9985   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1928   0.1736   0.1626   0.1833
## Detection Prevalence   0.2849   0.1939   0.1751   0.1629   0.1833
## Balanced Accuracy      0.9997   0.9977   0.9969   0.9960   0.9986
```

The final model has a training set accuracy of 99.51%, and an out of sample (test set) accuracy of 99.68%, which is quite good. The other details about the model, and the metrics are all presented above.  

Now let us finally get to predicting on our test dataset.

## Test Data

First of all, we need to reduce our test dataset to the same variables that our training data set contained.  


```r
origin2 <- read.csv("pml-testing.csv")
testing <- origin2[,names(df)[-55]] # last column df is classe, not in test data
# converting to numeric according to training set
for(i in 2:54){
     testing[,i] <- extract_numeric(testing[,i]) # convert factors to numeric
}
# converting first column to 2 levels as it should be
levels(testing[,1]) <- c("no","yes")
```
  
Now that the test dataset is similar in features to our training data set, let us predict using our model.   

```r
prediction <- predict(finalFit,testing)
pander(data.frame(ID = 1:20,Class = prediction))
```



|  ID  |  Class  |
|:----:|:-------:|
|  1   |    B    |
|  2   |    A    |
|  3   |    B    |
|  4   |    A    |
|  5   |    A    |
|  6   |    E    |
|  7   |    D    |
|  8   |    B    |
|  9   |    A    |
|  10  |    A    |
|  11  |    B    |
|  12  |    C    |
|  13  |    B    |
|  14  |    A    |
|  15  |    E    |
|  16  |    E    |
|  17  |    A    |
|  18  |    B    |
|  19  |    B    |
|  20  |    B    |

Using these predictions on the quiz resulted in 20/20.  

