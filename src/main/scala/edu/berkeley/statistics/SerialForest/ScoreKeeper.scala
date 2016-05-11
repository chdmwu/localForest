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

import scala.math.pow

abstract class ScoreKeeper {
  def moveLeft(yVal: Double)
  def getCurrentScore(): Double
  def reset()
}

class AnovaScoreKeeper(initialRightValues: IndexedSeq[Double]) extends ScoreKeeper {
  private val initialRightSum: Double = initialRightValues.sum
  private val initialRightNum: Int = initialRightValues.length

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
      0.0
    } else {
      -1 * pow(leftSum, 2) / leftNum - pow(rightSum, 2) / rightNum;
    }
  }

  def reset() = {
    rightSum = initialRightSum
    rightNum = initialRightNum
    leftSum = 0.0
    leftNum = 0
  }
}