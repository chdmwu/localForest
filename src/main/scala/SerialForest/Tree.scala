package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint

class Tree (head: Node) extends Serializable {
  def predict(testPoint: Vector[LabeledPoint]): Double = ???
  def getTopPNNs(testPoint: Vector[LabeledPoint]): Array[Long] = ???
}

object Tree {
  def train(input: Array[LabeledPoint]): Tree = ???
}
