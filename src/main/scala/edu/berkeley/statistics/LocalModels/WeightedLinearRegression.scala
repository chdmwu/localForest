package edu.berkeley.statistics.LocalModels

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.IndexedSeq

//TODO: Deal with singular covariance matrices
object WeightedLinearRegression {
  def getBetaHat(covMatrices: IndexedSeq[DenseMatrix[Double]],
                    crossCovArrays: IndexedSeq[DenseVector[Double]]): DenseVector[Double] = {

    covMatrices(0).cols match {
      case 0 => DenseVector(0)
      case _ => {
        val totCovMatrix = covMatrices.foldLeft[DenseMatrix[Double]](
          DenseMatrix.zeros[Double](covMatrices(0).rows, covMatrices(0).cols))
        {(prevMat: DenseMatrix[Double], newMat: DenseMatrix[Double]) => prevMat + newMat}
        val totCrossCovArrays = crossCovArrays.foldLeft[DenseVector[Double]](
          DenseVector.zeros[Double](crossCovArrays(0).length))
        {(prevVec: DenseVector[Double], newVec: DenseVector[Double]) => prevVec + newVec}

        totCovMatrix \ totCrossCovArrays
      }
    }
  }

  def getBetaHat(covMatrix: DenseMatrix[Double],
                 crossCovArray: DenseVector[Double]) : DenseVector[Double] = {
    covMatrix.cols match {
      case 0 => DenseVector(0)
      case _ => covMatrix \ crossCovArray
    }

  }

  def getFullMatrices(covMatrices: IndexedSeq[DenseMatrix[Double]],
                 crossCovArrays: IndexedSeq[DenseVector[Double]]): (DenseMatrix[Double], DenseVector[Double]) = {
    val totCovMatrix = covMatrices.foldLeft[DenseMatrix[Double]](
      DenseMatrix.zeros[Double](covMatrices(0).rows, covMatrices(0).cols))
    {(prevMat: DenseMatrix[Double], newMat: DenseMatrix[Double]) => prevMat + newMat}
    val totCrossCovArrays = crossCovArrays.foldLeft[DenseVector[Double]](
      DenseVector.zeros[Double](crossCovArrays(0).length))
    {(prevVec: DenseVector[Double], newVec: DenseVector[Double]) => prevVec + newVec}

    (totCovMatrix, totCrossCovArrays)
  }

  def getCovarianceMatrix(pnnsAndWeights: IndexedSeq[(LabeledPoint, Double)],
                            centerPoint: mllibVector):
  (DenseMatrix[Double], DenseVector[Double]) = {

    //val centerPointVec = new DenseVector(0.0 +: centerPoint.toArray)
    val centerPointVec = new DenseVector(centerPoint.toArray :+ 0.0)

    def covarianceFoldFn(previousMatrices: (DenseMatrix[Double], DenseVector[Double]),
                         pointAndWeight: (LabeledPoint, Double)):
    (DenseMatrix[Double], DenseVector[Double]) = {
      pointAndWeight match {
        case (point, weight) => {
          previousMatrices match {
            case (previousMatrix, previousVector) => {
              //val pointVector = new DenseVector(1.0 +: point.features.toArray) - centerPointVec
              val pointVector = new DenseVector(point.features.toArray :+ 1.0) - centerPointVec
              val outerProduct: DenseMatrix[Double] = (pointVector * pointVector.t)
              val crossProd: DenseVector[Double] = (pointVector :* point.label)
              (previousMatrix + (outerProduct :* weight),
                  previousVector + (crossProd :* weight))
            }
          }
        }
      }
    }

    val numPredictors = centerPoint.size
    // Form the covariance matrix
    pnnsAndWeights.foldLeft[(DenseMatrix[Double], DenseVector[Double])](
      (DenseMatrix.zeros[Double](numPredictors + 1, numPredictors + 1),
          DenseVector.zeros[Double](numPredictors + 1)))(covarianceFoldFn)
  }

  def reduceCov(covMatrix: DenseMatrix[Double],
                crossCovArray: DenseVector[Double], features: IndexedSeq[Int]): (DenseMatrix[Double], DenseVector[Double]) = {
    val nCols = covMatrix.cols
    val r = (0 to nCols-1).filter(features.toSet) :+ (nCols-1) //add in the last column since it is the intercept term
    (covMatrix(r, r).toDenseMatrix, crossCovArray(r).toDenseVector)
  }
}