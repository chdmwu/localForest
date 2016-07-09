/**
 * Copyright 2015 Adam Bloniarz, Christopher Wu, Ameet Talwalkar
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
      def getScoreAndSplitPoint(featureIndex: Int): (Double, Double) = {

        // Copy out the values of the column in this node
        val predictorValues = rowsHere.map(trainingData(_).features(featureIndex))
<<<<<<< HEAD

        // Get the indices of the sorted set of predictors
        // val predictorIndices = predictorValues.indices.sortBy(predictorValues(_))

=======
>>>>>>> bc2bb7146d667426f58aaba8f691ff2a0e210caa

        // Get the indices of the sorted set of predictors
        val predictorIndices = predictorValues.indices.toArray
        util.Sorting.quickSort[Int](predictorIndices)(
          Ordering.by[Int, Double](predictorValues(_)))

        var minScore: Double = 0.0

        // Reset the score keeper
        scoreKeeper.reset()

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

      val (bestScore, bestSplit, bestVariable) =
        (candidateVars.indices).view.foldLeft((currNodeScore, 0.0, -1))(
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

      if(bestVariable != -1){
        fit.updateGain(bestVariable, (currNodeScore - bestScore))
      } else {
        //println(rowsHere.length)
        //println(rowsHere.map(trainingData(_)))
        //scoresAndSplits.map(x => println(x._1 + " " + x._2))
        //println(candidateVars)
        //println(treeParameters.mtry)
        return makeLeafOrSplitRandomly //TODO: Chris Fix this hack somehow I guess
      }



      //println(currNodeScore - bestScore)
      // No viable split is found, so make a leaf
<<<<<<< HEAD
      if (bestScore == currNodeScore) { //changed default behavior of Scorekeeper, no longer returns 0.0 on no split.
=======
      if ((bestScore == 0.0) || (bestVariable == -1)) {
>>>>>>> bc2bb7146d667426f58aaba8f691ff2a0e210caa
        return makeLeafOrSplitRandomly
        //TODO: ADAM (from chris)  I dont know what this does but it doesnt work, crashes if no good split is found
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