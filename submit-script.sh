#!/bin/bash

spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --master local[8] \
    target/scala-2.10/DistributedForest-assembly-1.0.jar \
     Friedman1 1 10000 700 1000
#    GaussianProcess 4 50000 1000 200
    # Arguments are as follows
    # Simulation name
    # Number of partitions
    # Number of training points per partition
    # Number of PNNs per partition
    # Number of test points
