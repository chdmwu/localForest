package edu.berkeley.statistics.Simulations.DataGenerators

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.math._

/**
 * Created by christopher on 2/28/16.
 */
case class LinearGenerator(numTruePredictors : Int, numNonsensePredictors: Int, beta: Array[Double]) extends SimulationDataGenerator {

  def generateData(numObservations: Int, noiseSd: Double,
                   rng: scala.util.Random): IndexedSeq[LabeledPoint] = {
    def getSingleObservation: LabeledPoint = {
      val features = Array.fill(numTruePredictors)(rng.nextDouble - .5)
      val noise = Array.fill(numNonsensePredictors)(rng.nextDouble - .5)
      val outcome = features.zip(beta).map(x => x._1*x._2).sum + rng.nextGaussian * noiseSd
      /** val outcome = 4*features(0) + 2*features(1) + 8*features(2)+
        10 * features(3) + 5 * features(4) + rng.nextGaussian * noiseSd*/
      new LabeledPoint(outcome, Vectors.dense(features ++ noise))
    }
    IndexedSeq.fill(numObservations)(getSingleObservation)
  }
}