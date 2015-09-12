package DistributedForest

import SerialForest.{TreeParameters, RandomForestParameters, RandomForest}
import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vector => mllibVector}


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

  // TODO(adam): consider making this RDD[IndexedSeq[LabeledPoint]] to avoid copying
  // TODO(adam): make it so you can specify size of resample
  def train(trainData: RDD[LabeledPoint]): RDD[RandomForest] = {
    trainData.mapPartitions[RandomForest](x => Iterator(RandomForest.train(x.toIndexedSeq,
      RandomForestParameters(100, true, TreeParameters(3, 10)))))
  }

  def predictWithLocalModel(testData: IndexedSeq[mllibVector], forests: RDD[RandomForest],
                            numPNNsPerPartition: Int): IndexedSeq[Double] = {
    testData.map(testPoint => {
      val pnnsAndWeights = forests.flatMap((forest: RandomForest) =>
      {
        val rawPNNsAndWeights = forest.getTopPNNsAndWeights(testPoint, numPNNsPerPartition)
        rawPNNsAndWeights.map{ case (trainingPoint: LabeledPoint, weight: Double) => {
          val rawPNN = trainingPoint.features.toArray
          // Center the PNN on the test point
          val shiftedPNN = rawPNN.indices.map(idx => rawPNN(idx) - testPoint(idx))
          (new LabeledPoint(trainingPoint.label, Vectors.dense(shiftedPNN.toArray)), weight)
        }
        }
      }).collect()

      WeightedLinearRegression.betaHat(pnnsAndWeights)(0)
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
