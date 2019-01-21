SCRIPT=lr.py
SCRIPTCLEAN=lr_clean.py
LINT=--conf spark.extraListeners=com.groupon.sparklint.SparklintListener --packages com.groupon.sparklint:sparklint-spark201_2.11:1.0.8
OPT=
URL_MASTER=spark://MacBook-Air-di-Riccardo.local:7077
URL_DAV=spark://130.251.35.81:7077

script: 
	spark-submit $(OPT) $(SCRIPTCLEAN)

lint:
	spark-submit $(OPT) $(LINT) $(SCRIPTCLEAN)

1core: 
	spark-submit --master local $(LINT) $(SCRIPTCLEAN)


start_master:
	/spark/sbin/start-master.sh

stop_master:
	/spark/sbin/stop-master.sh

start_slave:
	/spark/sbin/start-slave.sh $(URL_MASTER)

stop_slave:
	/spark/sbin/stop-slave.sh $(URL_MASTER)

start_slave_dav:
	/spark/sbin/start-slave.sh $(URL_DAV)

stop_slave_dav:
	/spark/sbin/stop-slave.sh $(URL_DAV)

start_all: start_master start-slave start_slave_dav
stop_all: stop_master stop_slave stop_slave_dav


