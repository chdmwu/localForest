#!/bin/bash
export HADOOP_CONF_DIR=/etc/hadoop/conf
spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --deploy-mode client \
    --master yarn-client \
    --num-executors 8 \
    --conf "spark.executor.memory=7g" \
    localforest_jar/DistributedForest-assembly-1.0.jar \
    GaussianProcess 8 100000 1000 1000 100 10 30 300
#    Friedman1 8 100000 1000 1000 100 50
    # Arguments are as follows
    # Simulation name
    # Number of partitions
    # Number of training points per partition
    # Number of PNNs per partition
    # Number of test points
    # Batch size
    # (GaussianProcess simulation only) number of active dimensions
    # (GaussianProcess simulation only) number of inactive dimensions
    # (GaussianProcess simulation only) number of random basis functions
    # (GaussianProcess simulation only) number of trees in forest
    # (Friedman1 simulation only) number of inactive dimensions
