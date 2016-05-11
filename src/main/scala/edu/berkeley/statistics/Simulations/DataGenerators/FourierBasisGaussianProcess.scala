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
package edu.berkeley.statistics.Simulations.DataGenerators

import breeze.linalg.{sum, DenseVector, DenseMatrix, Axis}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.{tan, Pi, pow, exp, cos}

case class FourierBasisGaussianProcessFunction(
  basisCoefficients: DenseMatrix[Double],
  fourierCoefficients: DenseVector[Double]) {

  val numDimensions = basisCoefficients.cols

  def evaluate(x: DenseVector[Double]): Double = {
    val cosEvals: DenseVector[Double] = (basisCoefficients * x).map(cos(_))
    sum(cosEvals :* fourierCoefficients)
  }
}

object FourierBasisGaussianProcessFunction {
  def randomCauchy(rng: scala.util.Random): Double = tan(Pi * (rng.nextDouble - .5))

  def getFunction(numDimensions: Int, numBasisFunctions: Int,
                  sigma: Double, rng: scala.util.Random):
  FourierBasisGaussianProcessFunction = {
    val basisCoefficients = new DenseMatrix[Double](numBasisFunctions, numDimensions,
    Array.fill(numBasisFunctions * numDimensions)(randomCauchy(rng)))

    val colSums: DenseVector[Double] = sum(basisCoefficients :^ 2.0, Axis._1)
    val weights: DenseVector[Double] = colSums :* pow(sigma, 2.0)
    val fourierCoefficients: DenseVector[Double] = weights.map(x =>
      exp(-0.5 * x) * rng.nextGaussian())

    FourierBasisGaussianProcessFunction(basisCoefficients, fourierCoefficients)
  }
}

class FourierBasisGaussianProcessGenerator(function: FourierBasisGaussianProcessFunction,
                                              numNoiseVariables: Int)
    extends SimulationDataGenerator {
  def generateData(numObservations: Int, noiseSd: Double,
                   rng: scala.util.Random): IndexedSeq[LabeledPoint] = {
    // Generate the x values randomly
    IndexedSeq.fill(numObservations)({
      val x = Array.fill(function.numDimensions)(rng.nextDouble)
      val y = function.evaluate(new DenseVector(x)) + noiseSd * rng.nextGaussian
      new LabeledPoint(y, Vectors.dense(x ++ Array.fill(numNoiseVariables)(rng.nextGaussian)))
    })
  }
}
