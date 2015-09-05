package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint

abstract class Node private[SerialForest] ()

class InternalNode private[SerialForest] (leftChild: Node, rightChild: Node,
                                             splitVar: Int, spointPoint: Double)
    extends Node {
}

class LeafNode private (rowsHere: Vector[Long],
                trainingData: IndexedSeq[LabeledPoint]) extends Node {
}

object Node {
  // This will find the split point and do the splitting into child nodes
  def createNode(rowsHere: Vector[Long], treeParameters: TreeParameters,
                 trainingData: IndexedSeq[LabeledPoint],
                 rng: scala.util.Random): Node = {

    if (rowsHere.length <= treeParameters.nodeSize) {
      return new LeafNode(rowsHere, trainingData)
    } else {
      // Sample a set of variables to split on
      val numFeatures: Int = trainingData(0).features.size
      val candidate_vars = RandomSampling.
          sampleWithoutReplacement(numFeatures, treeParameters.mtry, rng)
    }

    // Place-holder for now
    new LeafNode(rowsHere, trainingData)
  }
}