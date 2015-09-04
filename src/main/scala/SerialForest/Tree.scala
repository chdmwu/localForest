package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint

class Tree private (head: Node) extends Serializable {
  def predict(testPoint: Vector): Double = ???
  def getTopPNNs(testPoint: Vector): Array[Long] = ???
}

object Tree {
  def train(input: Array[LabeledPoint]): Tree = ???
}
