#!/bin/bash

spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --master local[8] \
    target/scala-2.10/DistributedForest-assembly-1.0.jar \
     Friedman1 4 20000 1000000 1000 100 20
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
