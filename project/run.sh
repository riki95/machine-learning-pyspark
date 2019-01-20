#!/bin/bash

#spark-submit --master spark://davbuntu:7077 --conf spark.dynamicAllocation.enabled=false --conf spark.cores.max=1 main.py

#spark-submit --master local[1] --conf spark.executor.memory=450m main.py
#spark-submit --master local[*] --conf spark.default.cores=2 main.py
#spark-submit --master local[1] --conf spark.broadcast.blockSizes=100m main.py
#spark-submit --master local[1] --conf spark.files.maxPartitionBytes=1k main.py
#spark-submit --master local --conf spark.scheduler.mode=FAIR main.py

spark-submit --conf spark.extraListeners=com.groupon.sparklint.SparklintListener --packages com.groupon.sparklint:sparklint-spark201_2.11:1.0.8 lr.py

# spark-submit lr.py
