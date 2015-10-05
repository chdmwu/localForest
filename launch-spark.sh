#!/bin/bash
ADD_JARS="/home/hadoop/localforest_jar/DistributedForest-assembly-1.0.jar"
SPARK_MEM="7G"
spark-shell --driver-class-path "/home/hadoop/localforest_jar/DistributedForest-assembly-1.0.jar"
