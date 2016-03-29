package edu.berkeley.statistics.SerialForest

import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

class Tree private (head: Node, fit: FeatureImportance) extends Serializable {

  val featureImportance = fit
  private def getTerminalNode(testPoint: mllibVector): LeafNode = {
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
          return leaf
        }
      }
    }
    throw new IllegalStateException("Did not find leaf node; error in tree construction")
  }

  def predict(testPoint: mllibVector): Double = {
    getTerminalNode(testPoint).getValue()
  }

  def getPNNIndices(testPoint: mllibVector): IndexedSeq[Int] = {
    getTerminalNode(testPoint).rowsHere
  }
}

object Tree {
  def train(indices: IndexedSeq[Int],
            trainingData: IndexedSeq[LabeledPoint], treeParameters: TreeParameters,
            randomSeed: Long): Tree = {
    val fit = trainingData.length match {
      case 0 => new FeatureImportance(0)
      case _ => new FeatureImportance(trainingData(0).features.size)
    }

    new Tree(Node.createNode(indices,
      treeParameters, trainingData, new scala.util.Random(randomSeed), fit), fit)
  }
}
