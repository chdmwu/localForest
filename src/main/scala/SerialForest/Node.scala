package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint

abstract class Node private[SerialForest] () {
}

case class InternalNode private[SerialForest] (leftChild: Node, rightChild: Node,
                                          splitVar: Int, splitPoint: Double) extends Node {}

case class LeafNode private[SerialForest] (rowsHere: Vector[Int],
                        trainingData: IndexedSeq[LabeledPoint]) extends Node {
  def getValue(): Double = {
    rowsHere.foldLeft(0.0)(_ + trainingData(_).label) / rowsHere.length
  }
}

object Node {

  private def checkIfVariation(signal: IndexedSeq[Double]): Boolean = {
    // Check if there is any variation in the y values
    var previousItem: Double = signal(0)
    var foundVariation: Boolean = false

    signal.takeWhile(_ => foundVariation == false).foreach(item => {
      if (item != previousItem) {
        foundVariation = true
      } else {
        previousItem = item
      }
    })

    foundVariation
  }

  // This will find the split point and do the splitting into child nodes
  def createNode(rowsHere: Vector[Int], treeParameters: TreeParameters,
                 trainingData: IndexedSeq[LabeledPoint],
                 rng: scala.util.Random): Node = {

    if (rowsHere.length <= treeParameters.nodeSize) {
      LeafNode(rowsHere, trainingData)
    } else {
      // Sample a set of variables to split on
      val numFeatures: Int = trainingData(0).features.size
      val candidateVars = RandomSampling.
          sampleWithoutReplacement(numFeatures, treeParameters.mtry, rng)

      var minScore: Double = 0

      // Copy out the y values at this node
      val yValsAtNode = rowsHere.map(trainingData(_).label)

      // Check if there's any variation in the response signal
      if (checkIfVariation(yValsAtNode) == false) {
        return LeafNode(rowsHere, trainingData)
      }

      // Create initialized scoreKeeper
      val scoreKeeper: AnovaScoreKeeper = new AnovaScoreKeeper(yValsAtNode)

      // Function to get score and split point for a feature
      def getScoreAndSplitPoint(featureIndex: Int): (Double, Double) = {

        // Copy out the values of the column in this node
        val predictorValues = rowsHere.map(trainingData(_).features(featureIndex))

        // Check if there is any variation in this predictor
        if (checkIfVariation(predictorValues) == false) {
          return (0.0, 0.0)
        }

        // Get the indices of the sorted set of predictors
        val predictorIndices = (predictorValues.indices).sortWith{
          (index1, index2) => predictorValues(index1) < predictorValues(index2)
        }

        var minScore: Double = 0.0

        // Reset the score keeper
        scoreKeeper.reset()

        // TODO(adam) deal with ties
        predictorIndices.foldLeft((0.0, 0.0))(
          (bestScoreInfo, index) => {
            bestScoreInfo match {
              case (bestScore, bestSplit) =>
                scoreKeeper.moveLeft(yValsAtNode(index))
                val currentScore = scoreKeeper.getCurrentScore()
                if (currentScore < bestScore) {
                  (currentScore, predictorValues(index))
                } else {
                  (bestScore, bestSplit)
                }
            }
          })
      }

      val scoresAndSplits = candidateVars.map(getScoreAndSplitPoint(_))

      // TODO(adam) deal with ties and using midpoint as split
      val (bestScore, bestSplit, bestVariable) =
        (candidateVars.indices).foldLeft((0.0, 0.0, -1))(
          (bestScoreInfo, index) => {
            bestScoreInfo match {
              case (bestScore, bestSplit, bestPredictorIndex) =>
                if (scoresAndSplits(index)._1 < bestScore) {
                  (scoresAndSplits(index)._1, scoresAndSplits(index)._2, candidateVars(index))
                } else {
                  (bestScore, bestSplit, bestPredictorIndex)
                }
            }
          }
        )

      // No viable split is found, so make a leaf
      if (bestScore == 0.0) {
        return LeafNode(rowsHere, trainingData)
      }

      val (leftIndices, rightIndices) = rowsHere.partition(
        trainingData(_).features(bestVariable) <= bestSplit)

      if (leftIndices.length == 0 || rightIndices.length == 0) {
        System.out.println(leftIndices.length.toString + " " + rightIndices.length.toString)
        return LeafNode(rowsHere, trainingData)
      }

      InternalNode(Node.createNode(leftIndices, treeParameters, trainingData, rng),
        Node.createNode(rightIndices, treeParameters, trainingData, rng), bestVariable,
        bestSplit)

    }
  }
}