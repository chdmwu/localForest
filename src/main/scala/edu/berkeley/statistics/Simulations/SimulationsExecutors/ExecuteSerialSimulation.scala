package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.SerialForest._
import edu.berkeley.statistics.Simulations.DataGenerators.{FourierBasisGaussianProcessGenerator, FourierBasisGaussianProcessFunction, Friedman1Generator}
import edu.berkeley.statistics.Simulations.EvaluationMetrics
import scala.math.floor

object ExecuteSerialSimulation {
  def main(args: Array[String]) = {

    val simulationName = args(0)
    val nTrain: Int = args(1).toInt
    val nTest: Int = args(2).toInt

    // Generate the data
    val (dataGenerator, forestParameters) = simulationName match {
      case "Friedman1" => {
        val numNonsenseDimensions: Int = if (args.size > 3) args(3).toInt else 0
        (Friedman1Generator(numNonsenseDimensions),
            RandomForestParameters(100, true, TreeParameters(3, 10)))
      }
      case "GaussianProcess" => {
        val numActiveDimensions = if (args.size > 3) args(3).toInt else 20
        val numInactiveDimensions = if (args.size > 4) args(4).toInt else 0
        val numBasisFunctions = if (args.length > 5) args(5).toInt else numActiveDimensions * 500
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

    val trainingData = dataGenerator.generateData(nTrain, 3.0, scala.util.Random)
    val testData = dataGenerator.generateData(nTest, 0.0, scala.util.Random)

    println(trainingData(0).features.size)

    System.out.println("Training single tree")
    val singleTree = Tree.train(trainingData.indices, trainingData,
      forestParameters.treeParams, 0L)

    System.out.println("Getting predictions")
    val yHatTree = testData.map(x => singleTree.predict(x.features))

    System.out.println("RMSE is: " +
        EvaluationMetrics.rmse(yHatTree, testData.map(_.label)).toString)

    val t0 = System.currentTimeMillis()
    System.out.println("Training random forest")
    val forest = RandomForest.train(trainingData, forestParameters)
    System.out.println("Training time: " + (System.currentTimeMillis() - t0))
    val fit = forest.getFeatureImportance()
    //System.out.println(forest.getFeatureImportance().getNumberOfSplits)
    //System.out.println(forest.getFeatureImportance().getSplitGains)
    println(fit.getNumberOfSplits)
    //println(gains)
    println(fit.getAvgImportance)
    println(fit.getActiveSet())
    val ys =  trainingData.map(_.label);
    val variance = {
      val count = ys.length
      val mean =  ys.sum / count
      ys.map(y => (y - mean)*(y - mean)).sum
    }
    println(variance)
    println(ys.max)
    println(ys.min)
    println(ys.sum/ys.length)
    println(trainingData(0))
    println(FeatureImportance.getActiveFeatures(trainingData(0), fit.getActiveSet()))
    println(FeatureImportance.getActiveFeatures(trainingData(0), null))
    val yHatForest = testData.map(x => forest.predict(x.features))
    System.out.println("RMSE is: " +
        EvaluationMetrics.rmse(yHatForest, testData.map(_.label)).toString)
  }
}
