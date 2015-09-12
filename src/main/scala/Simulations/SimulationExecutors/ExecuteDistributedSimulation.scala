package Simulations.SimulationExecutors

import DistributedForest.DistributedForest
import Simulations.DataGenerators.Friedman1
import Simulations.EvaluationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

object ExecuteDistributedSimulation {
  def main(Args: Array[String]) = {

    val conf = new SparkConf().setAppName("DistributedForest").setMaster("local[6]")

    val sc = new SparkContext(conf)

    val numPartitions: Int = 6
    val numTrainingPointsPerPartition: Int = 10000
    val numPNNsPerPartition: Int = 700
    val numTestPoints: Int = 30

    System.out.println("Generating the data")
    // Generate the random seeds for simulating data
    val seeds = Array.fill(numPartitions)(scala.util.Random.nextLong())

    // Parallelize the seeds
    val seedsRDD = sc.parallelize(seeds, numPartitions)

    // Generate the data
    // TODO(adam) make this general so that any data generator can be used
    val trainingDataRDD = seedsRDD.flatMap(
      seed => Friedman1.generateData(
        numTrainingPointsPerPartition, 3.0, new scala.util.Random(seed)))

    // Train the forests
    System.out.println("Training forests")
    val forests = DistributedForest.train(trainingDataRDD)

    // Persist the forests in memory
    forests.persist()

    // Get the predictions
    System.out.println("Training local models and getting predictions")
    val (testPredictors, testLabels) = Friedman1.
        generateData(numTestPoints, 0.0, scala.util.Random).
        map(point => (point.features, point.label)).unzip

    val predictionsLocalModel = DistributedForest.predictWithLocalModel(
      testPredictors, forests, numPNNsPerPartition)

    val predictionsNaiveAveraging = DistributedForest.
        predictWithNaiveAverage(testPredictors, forests)

    sc.cancelAllJobs()
    sc.stop()

    def printMetrics(predictions: IndexedSeq[Double]): Unit = {
      System.out.println("RMSE is " + EvaluationMetrics.rmse(predictions, testLabels))
      System.out.println("Correlation is " + EvaluationMetrics.correlation(predictions, testLabels))
    }
    // Evaluate the predictions
    System.out.println("Performance using local model:")
    printMetrics(predictionsLocalModel)

    System.out.println("Performance using naive averaging")
    printMetrics(predictionsNaiveAveraging)
  }
}
