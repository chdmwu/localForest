package edu.berkeley.statistics.SerialForest

/**
 * Created by Adam on 9/11/15.
 */
object QuickSelect {

  def apply[A <% Ordered[A]](sequence: Array[A], n: Int, rng: scala.util.Random): A = {
    recursiveQuickSelect(sequence, n, 0, sequence.length, rng)
  }

  private def
  recursiveQuickSelect[A <% Ordered[A]](sequence: Array[A], n: Int, startIndex: Int, endIndex: Int,
                                                    rng: scala.util.Random): A = {

    val seqLength = endIndex - startIndex
    if (n > seqLength) {
      throw new IllegalArgumentException("Trying to get " + n +
          "th order statistic from sequence of length " + seqLength)
    }

    if (n == 1) {
      return (startIndex until endIndex).foldLeft(sequence(startIndex))((prev, elem) => {
        if (sequence(elem) > prev) sequence(elem) else prev
      })
    } else if (n == sequence.length) {
      return (startIndex until endIndex).foldLeft(sequence(startIndex))((prev, elem) => {
        if (sequence(elem) < prev) sequence(elem) else prev
      })
    }

    def swapElements(index1: Int, index2: Int): Unit = {
      if (index1 == index2) return
      val swapTmp = sequence(index2)
      sequence(index2) = sequence(index1)
      sequence(index1) = swapTmp
    }

    // Get a random pivot
    val pivotIndex = startIndex + rng.nextInt(seqLength)
    val pivotValue = sequence(pivotIndex)
    // Swap out pivotIndex with the endIndex
    swapElements(pivotIndex, endIndex - 1)
        // Partition in place
    var i: Int = startIndex - 1
    var j: Int = startIndex
    while (j < endIndex - 1) {
      if (sequence(j) <= pivotValue) {
        i += 1
        swapElements(i, j)
      }
      j += 1
    }
    swapElements(endIndex - 1, i + 1)
    //(startIndex until endIndex).foreach(x => System.out.print(sequence(x))); System.out.println()
    val numRight = endIndex - 1 - i
    if (numRight == n) {
      pivotValue
    } else if (numRight > n) {
      recursiveQuickSelect(sequence, n, i + 1, endIndex, rng)
    } else {
      recursiveQuickSelect(sequence, n - numRight, startIndex, i + 1, rng)
    }
  }
}
