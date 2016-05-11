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

import org.scalatest._

/**
 * Created by Adam on 9/11/15.
 */
class QuickSelectSpec extends FlatSpec {
  "QuickSelect" should "select the nth largest element" in {
    val testArray = Array[Int](6,1,5,2,4,3,0)
    assertResult(4) {
      QuickSelect(testArray, 3, scala.util.Random)
    }
  }

  it should "select the nth largest element when repeats are present" in {
    val testArray = Array[Int](6,1,5,2,4,3,3,0)
    assertResult(3) {
      QuickSelect(testArray, 4, scala.util.Random)
    }
    assertResult(3) {
      QuickSelect(testArray, 5, scala.util.Random)
    }
    assertResult(2) {
      QuickSelect(testArray, 6, scala.util.Random)
    }
  }

  it should "select the only element when given a length-one array" in {
    val testArray = Array[Int](1)
    assertResult(1) {
      QuickSelect(testArray, 1, scala.util.Random)
    }
  }

  it should "be able to select from a large list" in {
    val testArray = Array.fill(100000)(scala.util.Random.nextGaussian())

    val q = QuickSelect(testArray, 400, scala.util.Random)

    assert(q > 0)
  }

  it should "be able to select from a large list with repeats" in {
    val testArray = RandomSampling.sampleWithReplacement(100000, 100000, scala.util.Random)

    val q = QuickSelect(testArray, 50000, scala.util.Random)

    assert(q > 40000 && q < 60000)
  }
}
