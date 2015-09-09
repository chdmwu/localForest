package Simulations.FitSerialForest

import SerialForest.{TreeParameters, Tree}
import Simulations.DataGenerators.Friedman1
import scala.math.{pow, sqrt}

/**
 * Created by Adam on 9/9/15.
 */
object ExecuteSimulation {
  def main(args: Array[String]) = {

    val nTrain: Int = 100000
    val nTest: Int = 100

    val (trainingData, testData) = "Friedman1" match {
      case "Friedman1" => {
        (Friedman1.generateData(nTrain, 3.0, scala.util.Random),
            Friedman1.generateData(nTest, 0.0, scala.util.Random))
      }
    }

    System.out.println("Training single tree")
    val singleTree = Tree.train(trainingData, TreeParameters(8, 5, 0L))

    System.out.println("Getting predictions")
    val yHat = testData.map(x => singleTree.predict(x.features))

    val mse = yHat.indices.map(i => pow(yHat(i) - testData(i).label, 2)).
        foldLeft(0.0)(_ + _) / nTest
    val rmse = sqrt(mse)

    System.out.println("RMSE is: " + rmse.toString)
  }
}
