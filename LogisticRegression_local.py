#!/usr/bin/env python3
# coding: utf-8

# ## Classification Problem - Cross Validation - PySpark Local



### Import libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.session import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.context import SparkContext, SparkConf
from pipeline_tuning import DagCrossValidator


# ### Configure Spark
app_name = 'HPC Project'

### PySpark session initialization
conf = SparkConf().setAppName(app_name)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
# print(sc._conf.getAll())  # Get all the configuration parameters info


### Load the source data
csv = spark.read.csv('/hpc-data/bank_1g.csv', inferSchema=True, header=True, sep=',')


### Select features and label
data = csv.select(*(csv.columns[:-1]+ [((col("y")).cast("Int").alias("label"))]))
# print(data)


### Split the data and rename Y column
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")


### Define the pipeline
assembler = VectorAssembler(inputCols = data.columns[:-1], outputCol="features")
print("Input Columns: ", assembler.getInputCols())
print("Output Column: ", assembler.getOutputCol())

algorithm = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, algorithm])


### Tune Parameters
lr_reg_params = [0.01, 0.5, 2.0]
lr_elasticnet_param = [0.0, 0.5, 1.0]
lr_max_iter = [1,5,10]


### CrossValidation
folds = 5
parallelism = 9

evaluator=BinaryClassificationEvaluator()
paramGrid = ParamGridBuilder().addGrid(algorithm.regParam, lr_reg_params).addGrid(algorithm.maxIter, lr_max_iter).addGrid(algorithm.elasticNetParam, lr_elasticnet_param).build()

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=folds).setParallelism(parallelism)

#cv = DagCrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, parallelism=parallelism)


#### Training
import time
tic = time.time()

model = cv.fit(train)

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
input("Task completed. Close it with CTRL+C.")
