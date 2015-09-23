package edu.berkeley.statistics.Simulations.DataGenerators

import org.apache.spark.mllib.regression.LabeledPoint

trait SimulationDataGenerator extends Serializable {
  def generateData(numObservations: Int, noiseSd: Double,
                   rng: scala.util.Random): IndexedSeq[LabeledPoint]
}