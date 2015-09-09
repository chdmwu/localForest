package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector => mllibVector}

class Tree private (head: Node) extends Serializable {
  def predict(testPoint: mllibVector): Double = {
    var currentNode: Node = head
    while (true) {
      currentNode match {
        case internal: InternalNode => {
          if (testPoint(internal.splitVar) <= internal.splitPoint) {
            currentNode = internal.leftChild
          } else {
            currentNode = internal.rightChild
          }
        }
        case leaf: LeafNode => {
          return leaf.getValue()
        }
      }
    }
    0.0
  }

  def getTopPNNs(testPoint: mllibVector): Vector[Long] = ???
}

object Tree {
  def train(trainingData: IndexedSeq[LabeledPoint], treeParameters: TreeParameters): Tree = {
    return new Tree(Node.createNode((0 until trainingData.length).toVector,
      treeParameters, trainingData, new scala.util.Random(treeParameters.randomSeed)))
  }
}
