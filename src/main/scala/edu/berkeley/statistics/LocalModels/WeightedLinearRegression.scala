package edu.berkeley.statistics.LocalModels

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.{Vector => mllibVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

object WeightedLinearRegression extends WeightedLocalModel {
  private[LocalModels] def
  getCovarianceMatrix(pnnsAndWeights: Array[(LabeledPoint, Double)]): DenseMatrix[Double] = {
    val numPredictors = pnnsAndWeights(0)._1.features.size
    def covarianceFoldFn(previousMatrix: DenseMatrix[Double],
                         pointAndWeight: (LabeledPoint, Double)): DenseMatrix[Double] = {
      pointAndWeight match {
        case (point, weight) => {
          val pointMatrix = new DenseMatrix(numPredictors + 1, 1, 1.0 +: point.features.toArray)
          val outerProduct: DenseMatrix[Double] = (pointMatrix * pointMatrix.t)
          previousMatrix + (outerProduct :* weight)
        }
      }
    }
    // Form the covariance matrix
    pnnsAndWeights.foldLeft[DenseMatrix[Double]](
      DenseMatrix.zeros[Double](numPredictors + 1, numPredictors + 1))(covarianceFoldFn)
  }

  private[LocalModels] def
  getCrossProductVector(pnnsAndWeights: Array[(LabeledPoint, Double)]): DenseVector[Double] = {
    val numPredictors = pnnsAndWeights(0)._1.features.size
    def crossProdFoldFn(previousVector: DenseVector[Double],
                        pointAndWeight: (LabeledPoint, Double)): DenseVector[Double] = {
      pointAndWeight match {
        case (point, weight) => {
          val pointVector = new DenseVector(1.0 +: point.features.toArray)
          val crossProd: DenseVector[Double] = (pointVector :* point.label)
          previousVector + (crossProd :* weight)
        }
      }
    }

    pnnsAndWeights.foldLeft[DenseVector[Double]](
      DenseVector.zeros[Double](numPredictors + 1))(crossProdFoldFn)
  }

  def betaHat(pnnsAndWeights: Array[(LabeledPoint, Double)]): DenseVector[Double] = {
    val covarianceMatrix = WeightedLinearRegression.getCovarianceMatrix(pnnsAndWeights)
    val crossProductVector = WeightedLinearRegression.getCrossProductVector(pnnsAndWeights)
    covarianceMatrix \ crossProductVector
  }

  def fitAndPredict(trainingPointsAndWeights: Array[(LabeledPoint, Double)],
                    testPoint: mllibVector): Double = {
    val centeredTrainingData = centerTrainingData(trainingPointsAndWeights, testPoint)

    betaHat(centeredTrainingData)(0)
  }
}