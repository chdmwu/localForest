package edu.berkeley.statistics.LocalModels

import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by Adam on 9/16/15.
 */
trait WeightedLocalModel {
  def fitAndPredict(trainingPointsAndWeights: Array[(LabeledPoint, Double)],
                       testpoint: mllibVector)
}
