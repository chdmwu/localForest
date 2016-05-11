package edu.berkeley.statistics.Simulations

import scala.math._

object EvaluationMetrics {
  private def checkLengths(signal1: Seq[Double], signal2: Seq[Double]): Unit = {
    if (signal1.length != signal2.length) {
      throw new IllegalArgumentException(
        "Cannot calculate correlation between arrays of different length: " +
            signal1.length + " and " + signal2.length)
    }
  }
  
  def correlation(signal1: IndexedSeq[Double], signal2: IndexedSeq[Double]): Double = {
    checkLengths(signal1, signal2)

    val (num, signal1Norm, signal2Norm) = signal1.zip(signal2).foldLeft((0.0, 0.0, 0.0)){
      case ((numerator, signal1Norm, signal2Norm), (signal1Elem, signal2Elem)) => {
        (numerator + signal1Elem * signal2Elem, signal1Norm + pow(signal1Elem, 2),
            signal2Norm + pow(signal2Elem, 2))
      }}

    num / math.sqrt(signal1Norm * signal2Norm)
  }

  def rmse(signal1: IndexedSeq[Double], signal2: IndexedSeq[Double]) = {
    checkLengths(signal1, signal2)

    val mse = signal1.indices.map(i => pow(signal1(i) - signal2(i), 2)).sum / signal1.length
    sqrt(mse)
  }
}
