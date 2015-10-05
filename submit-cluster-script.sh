#!/bin/bash
export HADOOP_CONF_DIR=/etc/hadoop/conf
spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --deploy-mode client \
    --master yarn-client \
    --num-executors 10 \
    localforest_jar/DistributedForest-assembly-1.0.jar \
    Friedman1 12 5000 100000 1000 100 25
#    GaussianProcess 4 20000 1000 1000 50 10 30
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
    # (Friedman1 simulation only) number of inactive dimensions
