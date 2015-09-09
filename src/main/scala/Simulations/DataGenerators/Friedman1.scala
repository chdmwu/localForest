package Simulations.DataGenerators

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.math.{Pi, pow, sin}

trait SimulationDataGenerator {
  def generateData(numObservations: Int, noiseSd: Double,
                   rng: scala.util.Random): IndexedSeq[LabeledPoint]
}

object Friedman1 extends SimulationDataGenerator {
  def generateData(numObservations: Int, noiseSd: Double,
                   rng: scala.util.Random): IndexedSeq[LabeledPoint] = {
    def getSingleObservation: LabeledPoint = {
      val features = Array.fill(10)(rng.nextDouble)
      val outcome = 10 * sin(Pi * features(0) * features(1)) +
          20 * pow(features(2) - 0.5, 2) +
          10 * features(3) + 5 * features(4) + rng.nextGaussian * noiseSd
      new LabeledPoint(outcome, Vectors.dense(features))
    }
    IndexedSeq.fill(numObservations)(getSingleObservation)
  }
}