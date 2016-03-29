package edu.berkeley.statistics.SerialForest

import scala.math.pow

/**
 * Created by Adam on 9/8/15.
 */
abstract class ScoreKeeper {
  def moveLeft(yVal: Double)
  def getCurrentScore(): Double
  def reset()
}

class AnovaScoreKeeper(initialRightValues: IndexedSeq[Double]) extends ScoreKeeper {
  private val initialRightSum: Double = initialRightValues.sum
  private val initialRightNum: Int = initialRightValues.length
  private val sqSum: Double = initialRightValues.map(x => x*x).sum

  private var leftSum: Double = 0.0
  private var rightSum: Double = initialRightSum

  private var leftNum: Int = 0
  private var rightNum: Int = initialRightNum


  def moveLeft(yVal: Double) = {
    leftSum += yVal
    rightSum -= yVal
    leftNum += 1
    rightNum -= 1
  }

  def getCurrentScore(): Double = {
    if (leftNum == 0 || rightNum == 0) {
      -1 * pow(initialRightSum,2) / initialRightNum;
      //changed default behavior of Scorekeeper, no longer returns 0.0 on no split. Now returns score of parent node.
    } else {
      -1 * pow(leftSum, 2) / leftNum - pow(rightSum, 2) / rightNum;
    }
  }

  def getTotalNumPoints = initialRightNum

  def reset() = {
    rightSum = initialRightSum
    rightNum = initialRightNum
    leftSum = 0.0
    leftNum = 0
  }
}