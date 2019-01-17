#!/usr/bin/env python3
# coding: utf-8

# ## Classification Problem - Cross Validation - PySpark Local



### Import libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.session import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.context import SparkContext, SparkConf
import numpy as np


# ### Configure Spark
app_name = 'HPC Project'
cores_number = 'local[*]'  # 'local' for local and 'local[*] or local[n] for the number of cores to use'
master_thread = ''
cores = 4
memory = '8g'
storage_memory_cap = 1   # Default 0.6, this increase the storage memory cap


### PySpark session initialization
conf = SparkConf().setAppName(app_name).setMaster(master_thread).set('spark.driver.cores', cores).set('spark.memory.fraction', storage_memory_cap)
SparkContext.setSystemProperty('spark.executor.memory', memory)

sc = SparkContext(cores_number, conf=conf)
spark = SparkSession(sc)
# print(sc._conf.getAll())  # Get all the configuration parameters info


### Load the source data
csv = spark.read.csv('bank.csv', inferSchema=True, header=True, sep=',')


### Select features and label
data = csv.select(*(csv.columns[:-1]+ [((col("y")).cast("Int").alias("features"))]))
# print(data)


### Split the data and rename Y column
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("features", "trueLabel")


### Define the pipeline
assembler = VectorAssembler(inputCols = data.columns[:-1], outputCol="features")
print("Input Columns: ", assembler.getInputCols())
print("Output Column: ", assembler.getOutputCol())

layers = [4,5,4,3]

algorithm = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
# print(algorithm.explainParams())  # Explain LogisticRegression parameters

#### Training
import time
tic = time.time()

model = algorithm.fit(train)

toc = time.time()
print("Elapsed time ", toc-tic)


### Test the Model
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")
# print(*predicted.select('prediction', 'trueLabel').collect(), sep='\n')
print('true positives:', predicted.filter('trueLabel == 1').count())
print('true negatives:', predicted.filter('trueLabel == 0').count())


### Compute Confusion Matrix Metrics
tp = float(predicted.filter("prediction == 1.0 AND trueLabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND trueLabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND trueLabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND trueLabel == 1").count())
precision = tp / (tp + fp)
recall = tp / (tp + fn)
correctly_classified = (tp+tn) / predicted.count()
F1 = 2 * (precision * recall) / (precision + recall)

metrics = spark.createDataFrame([
					("TP", tp),
					("FP", fp),
					("TN", tn),
					("FN", fn),
					("Precision", precision),
					("Recall", recall),
					("Correctly Classified", correctly_classified),
                    ("F1", F1)
				],
				["metric", "value"])

### Print Results
metrics.show()


### Keep the PySpark Dahboard opened
# input("Task completed. Close it with CTRL+C.")
