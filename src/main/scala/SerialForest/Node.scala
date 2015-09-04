package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint

abstract class Node private[SerialForest] (private[SerialForest] val rowsHere: Array[Long],
                                           trainingData: Array[LabeledPoint])

class InternalNode private (rowsHere: Array[Long], trainingData: Array[LabeledPoint],
                    leftChild: Node, rightChild: Node) extends Node(rowsHere, trainingData) {
}

class LeafNode (rowsHere: Array[Long],
                trainingData: Array[LabeledPoint]) extends Node(rowsHere, trainingData) {

}

object InternalNode {
  // This will find the split point and do the splitting into child nodes
  def createInternalNode(rowsHere: Array[Long],
                         trainingData: Array[LabeledPoint]): InternalNode = ???
}