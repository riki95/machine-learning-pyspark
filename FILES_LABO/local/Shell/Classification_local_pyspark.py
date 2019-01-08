# Creating a Classification Model
#In this exercise, you will implement a classification model that uses features of a flight to predict whether or not the flight will be delayed.

#Import Spark SQL and Spark ML Libraries

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

# Load source data
# The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or features) for each flight, as well as a column indicating how many minutes late or early the flight arrived.

csv = spark.read.csv('../../data/flights.csv', inferSchema=True, header=True)

# Prepare the data
#Most modeling begins with exhaustive exploration and preparation of the data. In this example, the data has been cleaned for you. You will simply select a subset of columns to use as *features* and create a Boolean *label* field named **Late** with the value **1** for flights that arrived 15 minutes or more after the scheduled arrival time, or **0** if the flight was early or on-time.
data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 5).cast("Int").alias("Late")))

# Split the data in training and test
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows: ")
print(train_rows)
print(" Testing Rows: ")
print(test_rows)

# Prepare the training data
# To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the VectorAssembler class to transform the feature columns into a vector, and then rename the Late column to label.
assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
training = assembler.transform(train).select(col("features"), col("Late").alias("label"))

# Train a classification model
import time
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=1.)
tic = time.time()
model = lr.fit(training)
toc = time.time()
print("Elapsed time ", toc-tic)
print("Model trained!")

# Prepare the testing data
testing = assembler.transform(test).select(col("features"), col("Late").alias("trueLabel"))

# Test the model
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



