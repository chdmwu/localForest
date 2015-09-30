#!/bin/bash

spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteSerialSimulation \
    --master local[1] \
    target/scala-2.10/DistributedForest-assembly-1.0.jar \
    Friedman1 5000 700 
#    GaussianProcess 10000 1000 10 30
    # Arguments are as follows
    # Simulation name
    # Number of training points
    # Number of test points
    # (GaussianProcess simulation only) number of active dimensions
    # (GaussianProcess simulation only) number of inactive dimensions
    # (GaussianProcess simulation only) number of basis functions (optional) 
