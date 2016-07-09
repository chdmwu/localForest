import edu.berkeley.statistics.SerialForest.{FeatureImportance, RandomForestParameters, TreeParameters, RandomForest}


val bestFeatures = DistributedForest.validateActiveSet(validPredictors, validLabels, forests, nPnnsPerPartition, batchSize, fit)
val predictions = DistributedForest.predictWithLocalRegressionBatch(
      testPredictors, forests, nPnnsPerPartition, batchSize, fit.getTopFeatures(bestFeatures))


