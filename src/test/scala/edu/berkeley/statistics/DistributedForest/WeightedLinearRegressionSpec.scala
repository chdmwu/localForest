package edu.berkeley.statistics.DistributedForest

import breeze.linalg.{DenseMatrix, DenseVector}
import edu.berkeley.statistics.LocalModels.WeightedLinearRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest._

class WeightedLinearRegressionSpec extends FlatSpec {

  private def getNoiseFreeData: Array[(LabeledPoint)] = {
    // y = 3 + 2 * x0 + 4 * x1
    def getResponse(x: Array[Double]): Double = {
      3 + 2 * x(0) + 4 * x(1)
    }

    val xs = Array[Array[Double]](
      Array(-1, 5),
      Array(-5, 20),
      Array(4, 16)
    )

    xs.map(x => new LabeledPoint(getResponse(x), Vectors.dense(x)))
  }

  "WeightedLinearRegression" should "calculate the correct covariance matrix" in {
    // Test noise-free linear regression
    val trainingData = getNoiseFreeData

    val testCovarianceMat = WeightedLinearRegression.getCovarianceMatrix(
      trainingData.map(x => (x, 1.0)))

    val expectedCov: DenseMatrix[Double] = DenseMatrix(
      (3.0, -2.0, 41.0),
      (-2.0, 42.0, -41.0),
      (41.0, -41.0, 681.0))

    assert(testCovarianceMat == expectedCov)
  }

  "WeightedLinearRegression" should "calculate the correct cross-covariance vector" in {
    // Test noise-free linear regression
    val trainingData = getNoiseFreeData

    val testCrossVec = WeightedLinearRegression.getCrossProductVector(
      trainingData.map(x => (x, 1.0)))

    val expectedVec: DenseVector[Double] = DenseVector(169.0, -86.0, 2765.0)

    assert(testCrossVec == expectedVec)
  }

  "WeightedLinearRegression" should "calculate the correct least-squares regression vector" in {
    val trainingData = getNoiseFreeData

    val betaHat = WeightedLinearRegression.betaHat(trainingData.map(x => (x, 1.0)))

    val expectedBeta: DenseVector[Double] = DenseVector(3.0, 2.0, 4.0)

    assert(betaHat == expectedBeta)
  }
}
