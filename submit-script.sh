#!/bin/bash

spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --master local[8] \
    target/scala-2.10/DistributedForest-assembly-1.0.jar \
    GaussianProcess 4 50000 1000 1000 50 10 30
#     Friedman1 3 10000 700 1000 50
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
