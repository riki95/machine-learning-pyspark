# Let's consider here the implementation of a binary classifier using Logistic Regression and a fixed set of parameters to learn the model. We leverage the sklearn library

# import libraries
import sys
import scipy
import numpy as np
import matplotlib
import pandas
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

# Load the dataset
dataset = pandas.read_csv("../data/flights.csv")
print("Data set size", dataset.shape)

# Prepare the data
data = dataset[["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"]]
labels = (dataset[["ArrDelay"]]>5).astype(int)
print("Data matrix size ", data.shape)
print("Labels vector size",  labels.shape)

# Split the data
seed = 1234
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data.values, labels.values, test_size=0.3, random_state=seed)

print("Training set size:")
print("Data ", X_train.shape)
print("Labels ", Y_train.shape)
print("Test set size:")
print("Data ",  X_test.shape)
print("Labels ", Y_test.shape)

# Train a classification model
model = LogisticRegression(C=1.0,max_iter=10)
tic = time.time()
model.fit(X_train, Y_train.ravel())
toc = time.time()
print("Elapsed time ", toc-tic)

# Evaluate the model on the test set
predicted = model.predict(X_test)
c = confusion_matrix(Y_test, predicted)
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
