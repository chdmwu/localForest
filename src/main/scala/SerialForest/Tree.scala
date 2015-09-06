package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector



class Tree private (head: Node) extends Serializable {
  def predict(testPoint: Vector): Double = ???
  def getTopPNNs(testPoint: Vector): Vector[Long] = ???
}

object Tree {
  def train(trainingData: IndexedSeq[LabeledPoint], treeParameters: TreeParameters): Tree = {
    return new Tree(Node.createNode((0L until trainingData.length).toVector,
      treeParameters, trainingData, new scala.util.Random(0L))) // TODO(adam) deal with random seed
  }
}
