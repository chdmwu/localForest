#SILO: Supervised Local modeling method for distributing random forests with feature selection
This code implements the method introduced in

Bloniarz, A., Wu, C., Yu, B., & Talwalkar, A. (2016). Supervised Neighborhoods for Distributed Nonparametric Regression. In *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics* (pp. 1450-1459). [(link)](http://www.jmlr.org/proceedings/papers/v51/bloniarz16.pdf)

To compile and assemble jar, run `sbt assembly`. This will create `./target/scala-2.10/DistributedForest-assembly-1.0.jar`

##Basic usage
```scala
import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{FeatureImportance, RandomForestParameters, TreeParameters, RandomForest}

// Set random forest parameters
val forestParameters = RandomForestParameters(100,                    // Number of trees
                                              true,                   // Resample with replacement?
                                              TreeParameters(3,       // mtry
                                                             10))     // max number of training points in leaf node

// Train the models
// assumes trainingDataRDD is a RDD[LabeledPoint]
val forests = DistributedForest.train(trainingDataRDD, forestParameters)

// persist the trained forests in memory
forests.persist()

// set batch size for processing test points
val batchSize = 100

// set parameter to limit the size of supervised neighborhoods
// set to a large value to use the entire neighborhood (recommended)
val numPNNsPerPartition = 100000  

// get feature importance information from model
val fit = forests.getFeatureImportance()

// get best feature set using validation
// assumes validData is an IndexedSeq[Vector]
val bestFeatures = DistributedForest.validateActiveSet(validPredictors, validLabels, forests, nPnnsPerPartition, batchSize, fit)

// Make predictions at test points with local regression
// assumes testData is an IndexedSeq[Vector]
val predictions = DistributedForest.predictWithLocalRegressionBatch( testPredictors, forests, nPnnsPerPartition, batchSize, fit.getTopFeatures(bestFeatures))

```


