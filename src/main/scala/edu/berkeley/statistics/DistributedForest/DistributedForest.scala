package edu.berkeley.statistics.DistributedForest

import edu.berkeley.statistics.LocalModels.{WeightedLocalModel, WeightedLinearRegression}
import edu.berkeley.statistics.SerialForest.{TreeParameters, RandomForestParameters, RandomForest}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vector => mllibVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


/**
 * Creates a distributed localforest.
 *
 * Reads headerless csvs for training and testing data, assumes the last column is the response variable.
 */
object DistributedForest {
  val nPartitions = 5
  val numIterations = 100
  def main(args: Array[String]) = {
    var trainFile = "/home/christopher/train.csv"
    var testFile = "/home/christopher/test.csv"
    if(args.length==2) {
      val trainFile = args(0)
      val testFile = args(1)
      println("using training file: " + trainFile)
      println("using testing file: " + testFile)
    } else {
      println("Using default filenames, please input filename as commandline argument")
    }
    val conf = new SparkConf().setAppName("DistributedForest")
        .setMaster("local[4]")
    val sc = new SparkContext(conf)
    val trainData = sc.textFile(trainFile,nPartitions).map(createLabeledPoint)
    //printList(trainData.take(5))
    //val randomForests = train(trainData)
  }

  def createLabeledPoint(line: String) : LabeledPoint = {
    val tokens = line.split(",").map(_.toDouble)
    return new LabeledPoint(tokens.last, Vectors.dense(tokens.dropRight(1)))
  }

  // TODO(adam): consider making this RDD[IndexedSeq[LabeledPoint]]
  // TODO(adam): make it so you can specify size of resample
  // TODO(adam): make RF parameters tunable
  def train(trainData: RDD[LabeledPoint],
               parameters: RandomForestParameters): RDD[RandomForest] = {
    trainData.mapPartitions[RandomForest](x => Iterator(RandomForest.train(x.toIndexedSeq,
      parameters)))
  }

  def predictWithLocalModel(testData: IndexedSeq[mllibVector], forests: RDD[RandomForest],
                            numPNNsPerPartition: Int,
                               localModel: WeightedLocalModel): IndexedSeq[Double] = {
    testData.map(testPoint => {
      val pnnsAndWeights = forests.flatMap(_.getTopPNNsAndWeights(testPoint, numPNNsPerPartition))
          .collect()

      localModel.fitAndPredict(pnnsAndWeights, testPoint)
    })
  }

  def predictWithNaiveAverage(testData: IndexedSeq[mllibVector],
                              forests: RDD[RandomForest]): IndexedSeq[Double] = {
    testData.map(testPoint => {
      val allPredictions = forests.map(forest =>
        forest.predict(testPoint)).collect()
      allPredictions.sum / allPredictions.length
    })
  }
}
