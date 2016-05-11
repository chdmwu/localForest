package edu.berkeley.statistics.DistributedForest

import breeze.linalg.{DenseMatrix, DenseVector}
import edu.berkeley.statistics.LocalModels.WeightedLinearRegression
import edu.berkeley.statistics.SerialForest.{RandomForest, RandomForestParameters}
import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object DistributedForest {
  val nPartitions = 5
  val numIterations = 100

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
      (currentIndex until (currentIndex + batchSize)).foreach(i => {
        predictions(i) = predictionsBatch(i - currentIndex)
      })
      batchIndex += 1
      currentIndex += batchSize
    }

    predictions
  }

  def predictWithNaiveAverageBatch(testData: IndexedSeq[mllibVector],
                                   forests: RDD[RandomForest], batchSize: Int):
  IndexedSeq[Double] = {
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
