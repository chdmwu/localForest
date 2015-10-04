package edu.berkeley.statistics.DistributedForest

import breeze.linalg.{DenseMatrix, DenseVector}
import edu.berkeley.statistics.LocalModels.WeightedLinearRegression
import edu.berkeley.statistics.SerialForest.{RandomForest, RandomForestParameters}
import org.apache.spark.mllib.linalg.{Vector => mllibVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


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

  def predictWithLocalRegressionBatch(testData: IndexedSeq[mllibVector], forests: RDD[RandomForest],
                                          numPNNsPerPartition: Int,
                                          batchSize: Int): IndexedSeq[Double] = {
    val batchData = testData.grouped(batchSize).toIndexedSeq //group into batches
    val predictions = Array.fill(testData.size)(0.0)
    var batchIndex = 0
    var currentIndex = 0

    val nCols = testData(0).size + 1

    while (batchIndex < batchData.length) {
      val testDataBroadcasted = forests.context.broadcast(batchData(batchIndex))
      val covMatsForestsRaw = forests.map((f: RandomForest) => {
        val pnnsAndWeights = testDataBroadcasted.value.map(
          f.getTopPNNsAndWeights(_, numPNNsPerPartition))
        val pnnsWeightsAndTestPoint = pnnsAndWeights.zip(testDataBroadcasted.value)
        val outputMatrices = pnnsWeightsAndTestPoint.map(x =>
          WeightedLinearRegression.getCovarianceMatrix(x._1, x._2))
        val outputArrays: IndexedSeq[(Array[Double], Array[Double])] = outputMatrices.map{
          case (covMat: DenseMatrix[Double], crossCovVec: DenseVector[Double]) => {
            (covMat.valuesIterator.toArray, crossCovVec.valuesIterator.toArray)
          }
        }
        outputArrays
      }).collect

      val covMatsForests = covMatsForestsRaw.map(_.map{
        case (covMatRaw: Array[Double], crossCovVecRaw: Array[Double]) => {
          (new DenseMatrix[Double](nCols, nCols, covMatRaw),
              new DenseVector[Double](crossCovVecRaw))
        }
      })

      val predictionsBatch: IndexedSeq[Double] = batchData(batchIndex).indices.map((i: Int) => {
        val (partitionCovMats, partitionCrossCovs) = covMatsForests.map(_(i)).unzip
        val betaHat: DenseVector[Double] =
          WeightedLinearRegression.getBetaHat(partitionCovMats, partitionCrossCovs)
        betaHat(0)
      })
      // Fill the predictions
      // Ugh
      (currentIndex until (currentIndex + batchSize)).foreach(i => {
        predictions(i) = predictionsBatch(i - currentIndex)
      })
      batchIndex += 1
      currentIndex += batchSize
    }

    // Is this terrible?
    predictions
  }

  def predictWithNaiveAverageBatch(testData: IndexedSeq[mllibVector],
                                       forests: RDD[RandomForest], batchSize: Int): IndexedSeq[Double] = {
    var batchData = testData.grouped(batchSize).toIndexedSeq
    val predictions = Array.fill(testData.size)(0.0)
    var batchIndex = 0
    var currentIndex = 0

    while (batchIndex < batchData.length) {
      val testDataBroadcasted = forests.context.broadcast(batchData(batchIndex))
      val forestPreds = forests.map(f => {
        testDataBroadcasted.value.map(f.predict(_))
      }).collect
      val predictionsBatch = testDataBroadcasted.value.indices.map(i => {
        forestPreds.map(_(i)).sum / forestPreds.size
      })
      (currentIndex until (currentIndex + batchSize)).foreach(i => {
        predictions(i) = predictionsBatch(i - currentIndex)
      })
      currentIndex = currentIndex + batchSize
      batchIndex = batchIndex + 1
    }
    predictions
  }
}
