package edu.berkeley.statistics.Simulations.DataGenerators

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.math.{Pi, pow, sin}

case class Friedman1Generator(numNonsensePredictors: Int) extends SimulationDataGenerator {
  def generateData(numObservations: Int, noiseSd: Double,
                   rng: scala.util.Random): IndexedSeq[LabeledPoint] = {
    def getSingleObservation: LabeledPoint = {
      val features = Array.fill(5 + numNonsensePredictors)(rng.nextDouble)
      val outcome = 10 * sin(Pi * features(0) * features(1)) +
          20 * pow(features(2) - 0.5, 2) +
          10 * features(3) + 5 * features(4) + rng.nextGaussian * noiseSd
     /** val outcome = 4*features(0) + 2*features(1) + 8*features(2)+
        10 * features(3) + 5 * features(4) + rng.nextGaussian * noiseSd*/
      new LabeledPoint(outcome, Vectors.dense(features))
    }
    IndexedSeq.fill(numObservations)(getSingleObservation)
  }
}