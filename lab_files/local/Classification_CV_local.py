# Using Cross Validation
# In this exercise, you will use cross-validation to optimize parameters for a classification model.

# Prepare the Data

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time

# Load the source data
dataset = pandas.read_csv("../../data/flights.csv")
data = dataset[["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"]]
labels = (dataset[["ArrDelay"]]>5).astype(int)
seed = 1234
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data.values, labels.values, test_size=0.3, random_state=seed)

# Tune parameters with K-fold cross validation
# Try to play with :
# - the number of possible values for the parameters
# - the number k of folds in the cross validation
# - number of jobs (for n_jobs=0 simply remove the parameter)
model = LogisticRegression()
grid={"C":[1, 10], "max_iter":[100]}
model_cv = GridSearchCV(model, grid, cv=10)
tic = time.time()
model_cv.fit(X_train, Y_train.ravel())
toc = time.time()
print("Elapsed time : ", toc-tic)

# Test the model
predicted = model_cv.predict(X_test)
print(sum(predicted==1))
c = confusion_matrix(Y_test.ravel(), predicted)
tp = c[1,1]
tn = c[0,0]
fp = c[0,1]
fn = c[1,0]

print("True Positive ", tp)
print("False Positive ", fp)
print("True Negative ", tn)
print("False Negative", fn)


precision = tp/(tp+fp)
recall = tp/(tp+fn)
print("Precision ", precision)
print("Recall ", recall)
