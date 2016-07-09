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