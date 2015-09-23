package edu.berkeley.statistics.LocalModels

import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

object WeightedAverage extends WeightedLocalModel {

  def fitAndPredict(trainingPointsAndWeights: Array[(LabeledPoint, Double)],
                    testPoint: mllibVector): Double = {
    val (numerator, denominator) = trainingPointsAndWeights.foldLeft((0.0, 0.0)){
      case ((numerator: Double, denominator: Double),
      (trainingPoint: LabeledPoint, weight: Double)) => {
        val yValue = trainingPoint.label
        (numerator + weight * yValue, denominator + weight)
      }}
    numerator / denominator
  }
}
