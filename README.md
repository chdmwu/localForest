#SILO: Supervised Local modeling method for distributing random forests
This code implements the method introduced in

Bloniarz, A., Wu, C., Yu, B., & Talwalkar, A. (2016). Supervised Neighborhoods for Distributed Nonparametric Regression. In *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics* (pp. 1450-1459). [(link)](http://www.jmlr.org/proceedings/papers/v51/bloniarz16.pdf)

To compile and assemble jar, run `sbt assembly`. This will create `./target/scala-2.10/DistributedForest-assembly-1.0.jar`

##Basic usage
```scala
import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{RandomForestParameters, TreeParameters}

// Set random forest parameters
val forestParameters = RandomForestParameters(100,                    // Number of trees
                                              true,                   // Resample with replacement?
                                              TreeParameters(3,       // mtry
                                                             10))     // max number of training points in leaf node

// Train the models
// assumes trainingDataRDD is a RDD[LabeledPoint]
val forests = DistributedForest.train(trainingDataRDD, forestParameters)

// Make predictions at test points
// assumes testData is an IndexedSeq[Array[Double]]
val predictions = testData.map(forest.predict(_))
```


