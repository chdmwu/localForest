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
package edu.berkeley.statistics.SerialForest

object RandomSampling {
  def sampleWithoutReplacement(n: Int, sampleSize: Int,
                               rng: scala.util.Random): Array[Int] = {
    // Knuth's algorithm
    var t: Int = 0
    var m: Int = 0

    var sample: Array[Int] = Array.fill(sampleSize)(0)

    while (m < sampleSize) {
      if ((n - t) * rng.nextDouble >= sampleSize - m) {
        t += 1
      } else {
        sample(m) = t
        t += 1
        m += 1
      }
    }
    sample
  }

  def sampleWithReplacement(n: Int, sampleSize: Int,
                            rng: scala.util.Random): Array[Int] = {
    Array.fill(sampleSize)(rng.nextDouble()).map(u => scala.math.floor(n * u).toInt)
  }
}

