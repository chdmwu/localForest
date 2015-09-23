package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.LocalModels.{WeightedAverage, WeightedLinearRegression}
import edu.berkeley.statistics.SerialForest.{RandomForestParameters, TreeParameters}
import edu.berkeley.statistics.Simulations.DataGenerators.{FourierBasisGaussianProcessFunction, FourierBasisGaussianProcessGenerator, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import org.apache.spark.{SparkConf, SparkContext}

import scala.math.floor

object ExecuteDistributedSimulation {
  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("DistributedForest").setMaster("local[6]")

    val sc = new SparkContext(conf)

    // Parse the command line parameters
    val simulationName: String = args(0)
    val numPartitions: Int = try { args(1).toInt } catch { case _ : Throwable => {
      System.err.println("Unable to parse num partitions: " + args(1) + " is not an integer")
      printUsageAndQuit()
      0
    }}
    // For large simulations, may have to edit /usr/local/etc/sbtopts on MacOS if using sbt
    // or add --executor-memory XXG commandline option if using spark-submit
    val numTrainingPointsPerPartition: Int =  try { args(2).toInt } catch { case _ : Throwable => {
      System.err.println("Unable to parse num training points per partition: " + args(2) +
          " is not an integer")
      printUsageAndQuit()
      0
    }}
    val numPNNsPerPartition: Int = try { args(3).toInt } catch { case _ : Throwable => {
      System.err.println("Unable to parse num PNNs per partition: " + args(3) +
          " is not an integer")
      printUsageAndQuit()
      0
    }}
    val numTestPoints: Int = try { args(4).toInt } catch { case _ : Throwable => {
      System.err.println("Unable to parse num test points: " + args(4) +
          " is not an integer")
      printUsageAndQuit()
      0
    }}

    System.out.println("Generating the data")
    // Generate the random seeds for simulating data
    val seeds = Array.fill(numPartitions)(scala.util.Random.nextLong())

    // Parallelize the seeds
    val seedsRDD = sc.parallelize(seeds, numPartitions)

    // Generate the data
    val (dataGenerator, forestParameters) = simulationName match {
      case "Friedman1" => (Friedman1Generator,
          RandomForestParameters(100, true, TreeParameters(3, 10)))
      case "GaussianProcess" => {
        val numActiveDimensions = 20
        val numInactiveDimensions = 20
        val numBasisFunctions = if (args.length > 4) args(5).toInt else numActiveDimensions * 500
        val function = FourierBasisGaussianProcessFunction.getFunction(
          numActiveDimensions, numBasisFunctions, 0.05, new scala.util.Random(2015L))
        val generator = new FourierBasisGaussianProcessGenerator(function, numInactiveDimensions)
        (generator, RandomForestParameters(
          100, true, TreeParameters(
            floor((numActiveDimensions + numInactiveDimensions) / 3).toInt, 10)))
      }
      case other => {
        throw new IllegalArgumentException("Unknown simulation name: " + simulationName)
      }
    }

    val trainingDataRDD = seedsRDD.flatMap(
      seed => dataGenerator.generateData(
        numTrainingPointsPerPartition, 3.0, new scala.util.Random(seed)))

    // Train the forests
    System.out.println("Training forests")
    val forests = DistributedForest.train(trainingDataRDD, forestParameters)

    // Persist the forests in memory
    forests.persist()

    // Get the predictions
    System.out.println("Training local models and getting predictions")
    val (testPredictors, testLabels) = dataGenerator.
        generateData(numTestPoints, 0.0, scala.util.Random).
        map(point => (point.features, point.label)).unzip

    val predictionsLocalRegression = DistributedForest.predictWithLocalModel(
      testPredictors, forests, numPNNsPerPartition, WeightedLinearRegression)

    val predictionsLocalConstant = DistributedForest.predictWithLocalModel(
      testPredictors, forests, numPNNsPerPartition, WeightedAverage)

    val predictionsNaiveAveraging = DistributedForest.
        predictWithNaiveAverage(testPredictors, forests)

    sc.cancelAllJobs()
    sc.stop()

    def printMetrics(predictions: IndexedSeq[Double]): Unit = {
      System.out.println("RMSE is " + EvaluationMetrics.rmse(predictions, testLabels))
      System.out.println("Correlation is " + EvaluationMetrics.correlation(predictions, testLabels))
    }
    // Evaluate the predictions
    System.out.println("Performance using local regression model:")
    printMetrics(predictionsLocalRegression)

    System.out.println("Performance using local constant model:")
    printMetrics(predictionsLocalConstant)

    System.out.println("Performance using naive averaging")
    printMetrics(predictionsNaiveAveraging)
  }

  def printUsageAndQuit(): Unit = {
    System.err.println(
      "Usage: ExecuteDistributedSimulation <simulationName> <numPartitions> " +
          "<numTrainingPointsPerPartition> <numPNNsPerPartition> <numTestPoints> " +
    "[Simulation specific parameters]")
    System.exit(1)
  }
}
