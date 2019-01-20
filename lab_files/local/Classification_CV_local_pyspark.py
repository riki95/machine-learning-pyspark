# Tuning Model Parameters
#In this exercise, you will optimise the parameters for a classification model.

#Prepare the Data

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

# Define the pipeline
assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, lr])

# Tune the parameters
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 1, 10]).addGrid(lr.maxIter, [100,10,5]).build()

cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, numFolds=10)

import time
tic = time.time()
model = cv.fit(train)
toc = time.time()
print("Elapsed time ", toc-tic)

# Test the model
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")

# Evaluate the model
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
