#!/usr/bin/env python
# coding: utf-8

# ## Tuning Model Parameters
# 
# In this exercise, you will optimise the parameters for a classification model.
# 
# ### Prepare the Data
# 
# First, import the libraries you will need and prepare the training and test data:

# In[1]:


# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

# Load the source data
csv = spark.read.csv('../../data/flights.csv', inferSchema=True, header=True)

# Select features and label
data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 5).cast("Int").alias("label")))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")


# ### Define the Pipeline
# Now define a pipeline that creates a feature vector and trains a classification model

# In[2]:


# Define the pipeline
assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, lr])


# ### Tune Parameters
# You can tune parameters to find the best model for your data. A simple way to do this is to use  **TrainValidationSplit** to evaluate each combination of parameters defined in a **ParameterGrid** against a subset of the training data in order to find the best performing parameters.

# In[6]:


paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 1, 10]).addGrid(lr.maxIter, [100,10,5]).build()

cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, numFolds=10)

import time
tic = time.time()
model = cv.fit(train)
toc = time.time()
print("Elapsed time ", toc-tic)


# ### Test the Model
# Now you're ready to apply the model to the test data.

# In[4]:


prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")


# ### Compute Confusion Matrix Metrics
# Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
# - True Positives
# - True Negatives
# - False Positives
# - False Negatives
# 
# From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

# In[5]:


tp = float(predicted.filter("prediction == 1.0 AND trueLabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND trueLabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND trueLabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND trueLabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

