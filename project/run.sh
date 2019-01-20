#!/bin/bash

#spark-submit --conf spark.extraListeners=com.groupon.sparklint.SparklintListener --packages com.groupon.sparklint:sparklint-spark201_2.11:1.0.8 lr.py

# spark-submit lr.py

spark-submit lr_clean.py
