package SerialForest

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

