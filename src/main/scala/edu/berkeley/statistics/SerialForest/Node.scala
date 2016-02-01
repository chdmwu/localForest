package edu.berkeley.statistics.SerialForest

import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.floor

abstract class Node private[SerialForest] () {
}


case class InternalNode private[SerialForest] (leftChild: Node, rightChild: Node,
                                               splitVar: Int, splitPoint: Double) extends Node {}

case class LeafNode private[SerialForest] (rowsHere: IndexedSeq[Int],
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
  def createNode(rowsHere: IndexedSeq[Int], treeParameters: TreeParameters,
                 trainingData: IndexedSeq[LabeledPoint],
                 rng: scala.util.Random, fit: FeatureImportance): Node = {


    if (rowsHere.length <= treeParameters.nodeSize) {
      LeafNode(rowsHere, trainingData)
    } else {

      // Sample a set of variables to split on
      val numFeatures: Int = trainingData(0).features.size
      val candidateVars = RandomSampling.
          sampleWithoutReplacement(numFeatures, treeParameters.mtry, rng)

      // Copy out the y values at this node
      val yValsAtNode = rowsHere.map(trainingData(_).label)

      def makeLeafOrSplitRandomly: Node  = {
        if (rowsHere.size > treeParameters.nodeSize) {
          // A hack
          // Pick a random variable
          val splitVar = candidateVars(0)
          val predictorValues: IndexedSeq[Double] =
            rowsHere.map(trainingData(_).features(splitVar))
          val splitPoint = QuickSelect(predictorValues.toArray,
            floor(predictorValues.size / 2).toInt, rng)
          val (idcsLeft, idcsRight) =
            rowsHere.partition(trainingData(_).features(splitVar) <= splitPoint)
          if (idcsLeft.size == rowsHere.length || idcsRight.size == rowsHere.length) {
            if (idcsLeft.size > 1000) {
              System.out.println(idcsLeft.size)
            } else if (idcsRight.size > 1000){
              System.out.println(idcsRight.size)
            }
            LeafNode(rowsHere, trainingData)
          } else {
            InternalNode(Node.createNode(idcsLeft, treeParameters, trainingData, rng, fit),
              Node.createNode(idcsRight, treeParameters, trainingData, rng, fit), splitVar,
              splitPoint)
          }
        } else {
          LeafNode(rowsHere, trainingData)
        }
      }

      // Check if there is any variation in the response signal
      if (checkIfVariation(yValsAtNode) == false) {
        return makeLeafOrSplitRandomly
      }

      // Create initialized scoreKeeper
      val scoreKeeper: AnovaScoreKeeper = new AnovaScoreKeeper(yValsAtNode)
      val currNodeScore = scoreKeeper.getCurrentScore()

      // Function to get score and split point for a feature
      // TODO(adam): convert to using some and none for non-feasible variables?
      def getScoreAndSplitPoint(featureIndex: Int): (Double, Double) = {

        // Copy out the values of the column in this node
        val predictorValues = rowsHere.map(trainingData(_).features(featureIndex))

        // Get the indices of the sorted set of predictors
        // val predictorIndices = predictorValues.indices.sortBy(predictorValues(_))


        val predictorIndices = predictorValues.indices.toArray
        util.Sorting.quickSort[Int](predictorIndices)(
          Ordering.by[Int, Double](predictorValues(_)))

        //        java.util.Arrays.sort(predictorIndices map java.lang.Integer.valueOf,
        //          Ordering.by[java.lang.Integer, Double](predictorValues(_)))

        var minScore: Double = 0.0

        // Reset the score keeper
        scoreKeeper.reset()

        // TODO(adam) deal with ties
        var bestScore: Double = 0
        var bestSplit: Double = 0
        var foundPredictorVariation: Boolean = false

        // Handle the first iteration
        scoreKeeper.moveLeft(yValsAtNode(predictorIndices.head))
        bestScore = scoreKeeper.getCurrentScore()
        var lastPredictorValue: Double = predictorValues(predictorIndices.head)


        predictorIndices.tail.foreach(index => {
          scoreKeeper.moveLeft(yValsAtNode(index))
          val newPredictorValue = predictorValues(index)
          if (newPredictorValue != lastPredictorValue) {
            foundPredictorVariation = true
            val currentScore = scoreKeeper.getCurrentScore()
            if (currentScore < bestScore) {
              bestScore = currentScore
              bestSplit = newPredictorValue
            }
            lastPredictorValue = newPredictorValue
          }
        })



        if (!foundPredictorVariation) {
          (0.0, 0.0)
        } else {
          (bestScore, bestSplit)
        }
      }


      val scoresAndSplits = candidateVars.map(getScoreAndSplitPoint(_))


      // TODO(adam) deal with ties and using midpoint as split
      val (bestScore, bestSplit, bestVariable) =
        (candidateVars.indices).view.foldLeft((0.0, 0.0, -1))(
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

      val labelValues = rowsHere.map(trainingData(_).label)
      val pmean = labelValues.sum/labelValues.length
      val variance = labelValues.map(y => Math.pow(y-pmean,2)).sum/labelValues.length

      fit.updateGain(bestVariable, (currNodeScore - bestScore));
      //println(currNodeScore - bestScore)
      // No viable split is found, so make a leaf
      if (bestScore == currNodeScore) { //changed default behavior of Scorekeeper, no longer returns 0.0 on no split.
        return makeLeafOrSplitRandomly
      }

      val (leftIndices, rightIndices) = rowsHere.partition(
        trainingData(_).features(bestVariable) <= bestSplit)

      if (leftIndices.length == rowsHere.length || rightIndices.length == rowsHere.length) {
        return makeLeafOrSplitRandomly
      }

      InternalNode(Node.createNode(leftIndices, treeParameters, trainingData, rng, fit),
        Node.createNode(rightIndices, treeParameters, trainingData, rng, fit), bestVariable,
        bestSplit)
    }
  }
}