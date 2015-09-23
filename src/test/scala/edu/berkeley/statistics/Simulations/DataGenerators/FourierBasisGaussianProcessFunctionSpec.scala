package edu.berkeley.statistics.Simulations.DataGenerators

import breeze.linalg.DenseVector
import edu.berkeley.statistics.SerialForest.{TreeParameters, RandomForestParameters}
import org.scalatest._

import scala.math._

/**
 * Created by Adam on 9/18/15.
 */
class FourierBasisGaussianProcessFunctionSpec extends FlatSpec with Matchers {

  "FourierBasisGaussianProcessFunction" should "compute correct function values" in {
    val function = FourierBasisGaussianProcessFunction.getFunction(2, 3, 0.05,
      new scala.util.Random(2015L))

    function.evaluate(DenseVector(0.5, 0.5)) should be (-0.1565745 +- 1e-7)
    function.evaluate(DenseVector(0.25, 0.25)) should be (0.1360975 +- 1e-7)
    function.evaluate(DenseVector(0.3, 0.8)) should be (-0.4642075 +- 1e-7)
    function.evaluate(DenseVector(0.75, 0.9)) should be (-0.4416498 +- 1e-7)
  }

  "FourierBasisGaussianProcessGenerator" should "generate data with proper dimensions" in {
    val numActiveDimensions: Int = 20
    val numInactiveDimensions: Int = 20
    val function = FourierBasisGaussianProcessFunction.getFunction(
      numActiveDimensions, 500, 0.05, new scala.util.Random(2015L))
    val generator = new FourierBasisGaussianProcessGenerator(function, numInactiveDimensions)
    val data = generator.generateData(10, 5, scala.util.Random)
    assertResult(numActiveDimensions + numInactiveDimensions) {
      data(0).features.size
    }
  }
}
