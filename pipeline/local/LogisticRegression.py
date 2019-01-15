#!/usr/bin/env python3
# coding: utf-8

# ## Classification Problem - Cross Validation - PySpark Local

# ### Prepare the Data
# 
# First, import the libraries that we will need and prepare the training and test data:

# ### Import libraries

# In[1]:


from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.context import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

# ### Configure Spark

# In[2]:


master_threads = "local[1]"  # Local
# master_threads = "local[10]"  # Distributed

memory = '8g'
cores = 4
fraction = 0.6  # Default 0.6, this increase the storage memory cap


# ### PySpark session initialization

# In[3]:


conf = SparkConf().setAppName("HPC PySpark App").setMaster(master_threads).set('spark.driver.cores', cores).set('spark.memory.fraction', fraction)

SparkContext.setSystemProperty('spark.executor.memory', memory)
sc = SparkContext('local', conf=conf)
spark = SparkSession(sc)

# print(sc._conf.getAll())


# ### Load the source data

# In[4]:


# csv = spark.read.csv('../../data/bank.csv', inferSchema=True, header=True, sep=';')
csv = spark.read.csv('normalized2.csv', inferSchema=True, header=True, sep=',')


# In[5]:


### Select features and label


# In[6]:


data = csv.select(*(csv.columns[:-1]+ [((col("y")).cast("Int").alias("label"))]))
# print(data)


# In[7]:


### Split the data 70-30


# In[8]:


splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")


# ### Define the Pipeline
# Now define a pipeline that creates a feature vector and trains a classification model

# In[9]:


assembler = VectorAssembler(inputCols = data.columns[:-1], outputCol="features")
print("Input Columns: ", assembler.getInputCols())
print("Output Column: ", assembler.getOutputCol())

# print(lr.explainParams())
algorithm = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, algorithm])


# ### Tune Parameters
# You can tune parameters to find the best model for your data. A simple way to do this is to use  **TrainValidationSplit** to evaluate each combination of parameters defined in a **ParameterGrid** against a subset of the training data in order to find the best performing parameters.

# #### Regularization Params and iterations
# Save Regularization params and Max iteration in order to pass them at the paramGrid

# In[10]:


lr_reg_params = [0.1, 1, 10]
lr_max_iter = [100,10,5]


# #### CrossValidation
# Set CrossValidation estimator, evaluator, params and folds

# In[11]:


folds = 10
parallelism = 10
evaluator=BinaryClassificationEvaluator()
paramGrid = ParamGridBuilder().addGrid(algorithm.regParam, lr_reg_params).addGrid(algorithm.maxIter, lr_max_iter).build()

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=folds).setParallelism(parallelism)


# #### Training
# Starting and ending time for the fit

# In[12]:


import time
tic = time.time()

model = cv.fit(train)

toc = time.time()
print("Elapsed time ", toc-tic)


# ### Test the Model
# Now you're ready to apply the model to the test data.

# In[13]:


prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")
# print(*predicted.select('prediction', 'trueLabel').collect(), sep='\n')


# ### Compute Confusion Matrix Metrics
# Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
# - True Positives
# - True Negatives
# - False Positives
# - False Negatives
# 
# From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

# In[14]:


print('true positives:', predicted.filter('trueLabel == 1').count())
print('true negatives:', predicted.filter('trueLabel == 0').count())


# In[15]:


tp = float(predicted.filter("prediction == 1.0 AND trueLabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND trueLabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND trueLabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND trueLabel == 1").count())
correctly_classified = (tp+tn) / predicted.count()


# In[16]:


metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn)),
 ("Correctly Classified", correctly_classified)],["metric", "value"])
metrics.show()


# In[ ]:




