package edu.berkeley.statistics.LocalModels

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest._

class WeightedAverageSpec extends FlatSpec with Matchers {
  "WeightedAverage" should "calculate the correct weighted average" in {
    val testData = Array[(LabeledPoint, Double)](
      (LabeledPoint(3.0, Vectors.dense(Array.empty[Double])), 1.0),
      (LabeledPoint(4.0, Vectors.dense(Array.empty[Double])), 0.5),
      (LabeledPoint(5.0, Vectors.dense(Array.empty[Double])), 0.25)
    )

    WeightedAverage.fitAndPredict(
      testData, Vectors.dense(Array.empty[Double])) should be (3.571429 +- 0.000001)
  }
}
