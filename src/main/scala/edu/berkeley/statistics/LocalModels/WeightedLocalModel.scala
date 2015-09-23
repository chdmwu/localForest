package edu.berkeley.statistics.LocalModels

import org.apache.spark.mllib.linalg.{Vector => mllibVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by Adam on 9/16/15.
 */
abstract class WeightedLocalModel {
  def fitAndPredict(trainingPointsAndWeights: Array[(LabeledPoint, Double)],
                       testPoint: mllibVector): Double

  def centerTrainingData(trainingPointsAndWeights: Array[(LabeledPoint, Double)],
                         testPoint: mllibVector): Array[(LabeledPoint, Double)] = {
    trainingPointsAndWeights.map {
      case (trainingPoint: LabeledPoint, weight: Double) => {
        val rawPNN = trainingPoint.features.toArray
        // Center the PNN on the test point
        val shiftedPNN = rawPNN.indices.map(idx => rawPNN(idx) - testPoint(idx))
        (new LabeledPoint(trainingPoint.label, Vectors.dense(shiftedPNN.toArray)), weight)
      }
    }
  }
}