package edu.berkeley.statistics.Simulations.SimulationsExecutors

import edu.berkeley.statistics.SerialForest.{RandomForest, RandomForestParameters, Tree, TreeParameters}
import edu.berkeley.statistics.Simulations.DataGenerators.Friedman1
import edu.berkeley.statistics.Simulations.EvaluationMetrics

object ExecuteSerialSimulation {
  def main(args: Array[String]) = {

    val nTrain: Int = 50000
    val nTest: Int = 1000

    val (trainingData, testData) = "Friedman1" match {
      case "Friedman1" => {
        (Friedman1.generateData(nTrain, 3.0, scala.util.Random),
            Friedman1.generateData(nTest, 0.0, scala.util.Random))
      }
    }

    val treeParameters = TreeParameters(3, 10)
    System.out.println("Training single tree")
    val singleTree = Tree.train(trainingData.indices, trainingData, treeParameters, 0L)

    System.out.println("Getting predictions")
    val yHatTree = testData.map(x => singleTree.predict(x.features))

    System.out.println("RMSE is: " +
        EvaluationMetrics.rmse(yHatTree, testData.map(_.label)).toString)

    System.out.println("Training random forest")
    val forest = RandomForest.parTrain(trainingData,
      RandomForestParameters(100, true, treeParameters))

    val yHatForest = testData.map(x => forest.predict(x.features))
    System.out.println("RMSE is: " +
        EvaluationMetrics.rmse(yHatForest, testData.map(_.label)).toString)
  }
}
