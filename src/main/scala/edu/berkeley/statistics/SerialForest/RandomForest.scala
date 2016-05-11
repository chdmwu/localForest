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

import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

class RandomForest (trees: Seq[Tree], trainingData: IndexedSeq[LabeledPoint]) extends Serializable {

  private def getTopPNNIndicesAndWeights(testPoint: mllibVector,
                                         numPNNs: Int): IndexedSeq[(Int, Double)] = {
    val indexCounts = scala.collection.mutable.OpenHashMap.empty[Int, Double]

    trees.map(tree => {
      val indices = tree.getPNNIndices(testPoint)
      indices.foreach(idx => indexCounts.update(idx, indexCounts.get(idx).getOrElse(0.0) +
          1.0 / indices.size))
    })

    val totalWeight = indexCounts.values.sum

    if (numPNNs >= indexCounts.size) {
      indexCounts.mapValues(_ / totalWeight).toIndexedSeq
    } else {
      val weightCutoff = QuickSelect(indexCounts.values.toArray, numPNNs, scala.util.Random)
      indexCounts.filter(_._2 >= weightCutoff).mapValues(_ / totalWeight).toIndexedSeq
    }
  }

  def getTopPNNsAndWeights(testPoint: mllibVector,
                           numPNNs: Int): IndexedSeq[(LabeledPoint, Double)] = {
    this.getTopPNNIndicesAndWeights(testPoint, numPNNs).map{
      case (index: Int, weight: Double) => (trainingData(index), weight)}
  }

  def getTopPNNsAndWeightsBatch(testPoints: IndexedSeq[mllibVector],
                                numPNNs: Int): List[IndexedSeq[(LabeledPoint, Double)]] = {
    testPoints.map(getTopPNNsAndWeights(_, numPNNs)).toList
  }

  def predict(testPoint: mllibVector): Double = {
    trees.map(_.predict(testPoint)).sum / trees.length
  }
}

object RandomForest {
  private def getSamplingFn(parameters: RandomForestParameters):
  (Int, Int, scala.util.Random) => Array[Int] =
    if (parameters.replace) {
      RandomSampling.sampleWithReplacement
    } else {
      RandomSampling.sampleWithoutReplacement
    }

  def train(trainingData: IndexedSeq[LabeledPoint],
            parameters: RandomForestParameters): RandomForest = {
    val samplingFn = getSamplingFn(parameters)
    new RandomForest(List.fill(parameters.ntree)(scala.util.Random.nextLong).map(
      seed => Tree.train(
        samplingFn(trainingData.length, trainingData.length,
          new scala.util.Random(seed)),
        trainingData, parameters.treeParams, seed)), trainingData)
  }
}