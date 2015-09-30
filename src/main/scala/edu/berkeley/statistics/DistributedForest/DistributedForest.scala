package edu.berkeley.statistics.DistributedForest

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
    var batchData = testData.grouped(batchSize).toIndexedSeq //group into batches
    batchData.flatMap((batch: IndexedSeq[mllibVector]) => {
      val covMatsForests = forests.map((f: RandomForest) => {
        val pnnsAndWeights = batch.map(f.getTopPNNsAndWeights(_, numPNNsPerPartition))
        val pnnsWeightsAndTestPoint = pnnsAndWeights.zip(batch)
        pnnsWeightsAndTestPoint.map(x =>
          WeightedLinearRegression.getCovarianceMatrix(x._1, x._2))
      }).collect
      batch.indices.map((i: Int) => {
        val (partitionCovMats, partitionCrossCovs) = covMatsForests.map(_(i)).unzip
        WeightedLinearRegression.getBetaHat(partitionCovMats, partitionCrossCovs)(0)
      })
    })
  }

  def predictWithNaiveAverageBatch(testData: IndexedSeq[mllibVector],
                                   forests: RDD[RandomForest], batchSize: Int): IndexedSeq[Double] = {
    var batchData = testData.grouped(batchSize).toIndexedSeq
    batchData.flatMap(batch => {
      val forestPreds = forests.map(f => {
        batch.map(f.predict(_))
      }).collect
      batch.indices.map(i => {
        forestPreds.map(_(i)).sum / forestPreds.size
      })
    })
  }
}
