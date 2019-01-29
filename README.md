# Machine Learning on the Cloud

[![HitCount](http://hits.dwyl.io/riki_95/machine-learning-pyspark.svg)](http://hits.dwyl.io/riki_95/machine-learning-pyspark)

When do we need to compute Machine Learning tasks on the Cloud? This project wants to answer to this question.
We will study the PySpark architecture in Local creating our own cluster, then we will reproduce our experiments on the Cloud to see the differences.

## Getting Started

Download the repo:

```
git clone https://github.com/riki95/machine-learning-pyspark
```

Inside dataset_normalizer folder you can find the pandas code used to adapt the dataset.
Inside pipeline_tuning you find the [different Cross Validator](https://github.com/riki95/PipelineTuning) I have used in an experiment.

bank.csv is the 10MB dataset and can be also downloaded from here: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

There are 3 different files for the algorithm:
- LogisticRegression_local to be used in local and can also be used with the pipeline_tuning. In order to launch different tests you can use the MAKEFILE I have created to make it faster using Sparklint, which shows an interface like this to monitor the execution:

![Interface](https://i.imgur.com/xtEH0vh.png)

- LogisticRegression_GCP which should be used on Google Cloud Platform, it changes in folds and parallelism and also the csv read follows a cloud path.
- LogisticRegression_HDI is a Notebook file to run on Azure HDInsight Jupyter Notebook, just upload the dataset on HDI and run it (can also be run in GCP using Notebook if you change che path of the dataset)

## Author

* **Riccardo Basso** - Università degli studi di Genova - *High Performance Computing 2018-2019*
