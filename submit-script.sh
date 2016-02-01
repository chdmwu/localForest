#!/bin/bash
spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --master local[8] \
    target/scala-2.10/DistributedForest-assembly-1.0.jar \
	--simulationName $1 --nPartitions 4 --nPointsPerPartition 1000 --nPnnsPerPartition 500 --nTest 5000 --batchSize 100 --nActive 10 --nInactive $2 --runGlobalRF 1 --minNodeSize 25 --runOracle 1 --threshold1 .2 --threshold2 .4
	#    --simulationName GaussianProcess --nPartitions 4 --nPointsPerPartition 5000 --nPnnsPerPartition 500 --nTest 5000 --batchSize 100 --nActive 5 --nInactive $1 --runGlobalRF 1 --minNodeSize 25
#    GaussianProcess 4 2000 1000 1000 100 5 30
#Friedman1 4 2000 10000 1000 100 20
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
