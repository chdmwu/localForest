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
}
