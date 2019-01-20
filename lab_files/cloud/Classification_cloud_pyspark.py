#!/usr/bin/env python
# coding: utf-8

# ## Creating a Classification Model
# 
# In this exercise, you will implement a classification model that uses features of a flight to predict whether or not the flight will be delayed.
# 
# ### Import Spark SQL and Spark ML Libraries
# 
# First, import the libraries you will need:

# In[ ]:


from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


# ### Load Source Data
# The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or *features*) for each flight, as well as a column indicating how many minutes late or early the flight arrived.

# In[ ]:


csv = spark.read.csv('wasb:///data/flights.csv', inferSchema=True, header=True)


# ### Prepare the Data
# Most modeling begins with exhaustive exploration and preparation of the data. In this example, the data has been cleaned for you. You will simply select a subset of columns to use as *features* and create a Boolean *label* field named **Late** with the value **1** for flights that arrived 15 minutes or more after the scheduled arrival time, or **0** if the flight was early or on-time.

# In[ ]:


data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 5).cast("Int").alias("Late")))


# ### Split the Data
# It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.

# In[ ]:


splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows: ") 
print(train_rows)
print(" Testing Rows: ") 
print(test_rows)


# ### Prepare the Training Data
# To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **Late** column to **label**.

# In[ ]:


assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
training = assembler.transform(train).select(col("features"), col("Late").alias("label"))


# ### Train a Classification Model
# Next, you need to train a classification model using the training data. To do this, create an instance of the classification algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this exercise, you will use a *Logistic Regression* classification algorithm - though you can use the same technique for any of the classification algorithms supported in the spark.ml API.

# In[ ]:


import time
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=1.)
tic = time.time()
model = lr.fit(training)
toc = time.time()
print("Elapsed time ", toc-tic)
print("Model trained!")


# ### Prepare the Testing Data
# Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **Late** column to **trueLabel**.

# In[ ]:


testing = assembler.transform(test).select(col("features"), col("Late").alias("trueLabel"))


# ### Test the Model
# Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict delay status for flights where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted status to the actual status. 

# In[ ]:


prediction = model.transform(testing)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")

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

