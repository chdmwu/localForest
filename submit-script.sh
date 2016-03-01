#!/bin/bash
spark-submit \
    --class edu.berkeley.statistics.Simulations.SimulationsExecutors.ExecuteDistributedSimulation \
    --master local[8] \
    ./target/scala-2.10/DistributedForest-assembly-1.0.jar \
	--trainFile $1train.csv --validFile $1valid.csv --testFile $1test.csv --nPartitions 1 --nPnnsPerPartition 100000 --batchSize 100 --runGlobalRF 1 --minNodeSize 10
	
	
	#parameters accepted:

      #  case "--trainFile" :: value :: tail =>

      #  case "--validFile" :: value :: tail =>

      #  case "--testFile" :: value :: tail =>

      #  case "--nPartitions" :: value :: tail =>

      #  case "--nTrain" :: value :: tail =>

      #  case "--nPnnsPerPartition" :: value :: tail =>

      #  case "--nTest" :: value :: tail =>

      #  case "--batchSize" :: value :: tail =>

      #  case "--nActive" :: value :: tail =>

      #  case "--nInactive" :: value :: tail =>

       # case "--nBasis" :: value :: tail =>

       # case "--mtry" :: value :: tail =>

       # case "--ntree" :: value :: tail =>

       # case "--minNodeSize" :: value :: tail =>

       # case "--globalMinNodeSize" :: value :: tail =>

       # case "--runGlobalRF" :: value :: tail =>

       # case "--runOracle" :: value :: tail =>

       # case "--sampleWithReplacement" :: value :: tail =>

      #  case "--nValid" :: value :: tail =>

       # case "--normalizeLabel" :: value :: tail =>

      #  case "--runLinear" :: value :: tail =>

