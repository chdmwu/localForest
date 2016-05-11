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
package edu.berkeley.statistics.LocalModels

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

object WeightedLinearRegression {

  def getBetaHat(covMatrices: IndexedSeq[DenseMatrix[Double]],
                    crossCovArrays: IndexedSeq[DenseVector[Double]]): DenseVector[Double] = {
    val totCovMatrix = covMatrices.foldLeft[DenseMatrix[Double]](
      DenseMatrix.zeros[Double](covMatrices(0).rows, covMatrices(0).cols))
          {(prevMat: DenseMatrix[Double], newMat: DenseMatrix[Double]) => prevMat + newMat}
    val totCrossCovArrays = crossCovArrays.foldLeft[DenseVector[Double]](
      DenseVector.zeros[Double](crossCovArrays(0).length))
    {(prevVec: DenseVector[Double], newVec: DenseVector[Double]) => prevVec + newVec}

    totCovMatrix \ totCrossCovArrays
  }

  def getCovarianceMatrix(pnnsAndWeights: IndexedSeq[(LabeledPoint, Double)],
                            centerPoint: mllibVector):
  (DenseMatrix[Double], DenseVector[Double]) = {

    val centerPointVec = new DenseVector(0.0 +: centerPoint.toArray)

    def covarianceFoldFn(previousMatrices: (DenseMatrix[Double], DenseVector[Double]),
                         pointAndWeight: (LabeledPoint, Double)):
    (DenseMatrix[Double], DenseVector[Double]) = {
      pointAndWeight match {
        case (point, weight) => {
          previousMatrices match {
            case (previousMatrix, previousVector) => {
              val pointVector = new DenseVector(1.0 +: point.features.toArray) - centerPointVec
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
}