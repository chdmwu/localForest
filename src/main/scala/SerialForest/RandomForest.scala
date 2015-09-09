package SerialForest

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

class RandomForest (trees: List[Tree]) extends Serializable {
  def getPNNs(input: Vector): List[LabeledPoint] = ???
  def getWeights(input: Vector): List[Double] = ???
}

object RandomForest {
  def train(input: IndexedSeq[LabeledPoint]): IndexedSeq[RandomForest] = ???
}

