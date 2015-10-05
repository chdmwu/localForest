package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.DistributedForest.DistributedForest
import edu.berkeley.statistics.SerialForest.{RandomForestParameters, TreeParameters}
import edu.berkeley.statistics.Simulations.DataGenerators.{FourierBasisGaussianProcessFunction, FourierBasisGaussianProcessGenerator, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Strategy

import scala.math.floor

object ExecuteMLlibSimulation {
  def main(args: Array[String]) = {

    var argIndex = 0
    def incrementArgIndex = {
      val old = argIndex
      argIndex += 1
      old
    }

    val conf = new SparkConf().setAppName("DistributedForest")

    val sc = new SparkContext(conf)

    val numTrees = 100

    // Parse the command line parameters
    val simulationName: String = args(incrementArgIndex)
    val numPartitions: Int = try { args(incrementArgIndex).toInt } catch { case _ : Throwable => {
      System.err.println("Unable to parse num partitions: " + args(argIndex) + " is not an integer")
      printUsageAndQuit()
      0
    }}

    // For large simulations, may have to edit /usr/local/etc/sbtopts on MacOS if using sbt
    // or add --executor-memory XXG commandline option if using spark-submit
    val numTrainingPointsPerPartition: Int =  try { args(incrementArgIndex).toInt } catch {
      case _ : Throwable => {
        System.err.println("Unable to parse num training points per partition: " + args(argIndex) +
          " is not an integer")
        printUsageAndQuit()
        0
      }}
    val numTestPoints: Int = try { args(incrementArgIndex).toInt } catch {
      case _ : Throwable => {
        System.err.println("Unable to parse num test points: " + args(argIndex) +
          " is not an integer")
        printUsageAndQuit()
        0
      }}
    val minInstancesPerNodes: Int = try { args(incrementArgIndex).toInt } catch {
      case _ : Throwable => {
        System.err.println("Unable to parse minInstancesPerNodes: " + args(argIndex) +
          " is not an integer")
        printUsageAndQuit()
        0
      }}
    val maxDepth: Int = try { args(incrementArgIndex).toInt } catch {
      case _ : Throwable => {
        System.err.println("Unable to parse maxDepth: " + args(argIndex) +
          " is not an integer")
        printUsageAndQuit()
        0
      }}

    val maxBins: Int = try { args(incrementArgIndex).toInt } catch {
      case _ : Throwable => {
        System.err.println("Unable to parse maxBins: " + args(argIndex) +
          " is not an integer")
        printUsageAndQuit()
        0
      }}

    System.out.println("Generating the data")
    // Generate the random seeds for simulating data
    val seeds = Array.fill(numPartitions)(scala.util.Random.nextLong())

    // Parallelize the seeds
    val seedsRDD = sc.parallelize(seeds, numPartitions)

    var d = -1;

    // Generate the data
    val (dataGenerator, forestParameters) = simulationName match {
      case "Friedman1" => {
        val numNonsenseDimensions: Int = args(incrementArgIndex).toInt
        d = 5 + numNonsenseDimensions
        (Friedman1Generator(numNonsenseDimensions),
          RandomForestParameters(100, true, TreeParameters(3, 10)))

      }
      case "GaussianProcess" => {
        val numActiveDimensions = args(incrementArgIndex).toInt
        val numInactiveDimensions = args(incrementArgIndex).toInt
        d = numActiveDimensions + numInactiveDimensions
        val numBasisFunctions =
          if (args.length > argIndex) args(incrementArgIndex).toInt
          else numActiveDimensions * 500
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

    trainingDataRDD.persist()
    val forceTrainingData = trainingDataRDD.count()

    // Train the forests
    System.out.println("Training forests")
    val treeStrategy = Strategy.defaultStrategy("Regression")
    treeStrategy.setMinInstancesPerNode(minInstancesPerNodes)
    treeStrategy.setMaxDepth(maxDepth)
    treeStrategy.setMaxBins(maxBins)
    val featureSubsetStrategy = "auto" // Let the algorithm choose.

    val rfTrainStart = System.currentTimeMillis
    val model = RandomForest.trainRegressor(trainingDataRDD,
      treeStrategy, numTrees, featureSubsetStrategy, seed = 12345)
    val rfTrainTime = System.currentTimeMillis - rfTrainStart

    // Get the predictions
    System.out.println("Training local models and getting predictions")
    val (testPredictors, testLabels) = dataGenerator.
      generateData(numTestPoints, 0.0, scala.util.Random).
      map(point => (point.features, point.label)).unzip

    val mllibStart = System.currentTimeMillis
    val mllibPredictions = testPredictors.map(model.predict(_))
    val mllibTestTime = System.currentTimeMillis - mllibStart


    sc.cancelAllJobs()
    sc.stop()

    def printMetrics(predictions: IndexedSeq[Double]): Unit = {
      System.out.println("RMSE is " + EvaluationMetrics.rmse(predictions, testLabels))
      System.out.println("Correlation is " +
        EvaluationMetrics.correlation(predictions, testLabels))
    }

    System.out.println("Dataset: " + simulationName)
    System.out.println("Number of partitions: " + numPartitions)
    System.out.println("dimensions: " + d)
    System.out.println("Number of test points: " + numTestPoints)

    // Evaluate the predictions
    System.out.println("Performance using MLlib random forest:")
    printMetrics(mllibPredictions)


    System.out.println("Train time, MLlib: " + rfTrainTime)
    System.out.println("Test time, MLlib: " + mllibTestTime)

  }

  def printUsageAndQuit(): Unit = {
    System.err.println(
      "Usage: ExecuteMLlibSimulation <simulationName> <numPartitions> " +
        "<numTrainingPointsPerPartition>  <numTestPoints> " +
        "<minInstancesPerNode> <maxDepth> <maxBins> " +
        "[Simulation specific parameters]")
    System.exit(1)
  }
}
