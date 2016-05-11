/**
 * Copyright 2015 Adam Bloniarz, Christopher Wu, Ameet Talwalkar
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.berkeley.statistics.SerialForest

import org.apache.spark.mllib.linalg.{Vector => mllibVector}
import org.apache.spark.mllib.regression.LabeledPoint

class Tree private (head: Node) extends Serializable {
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
    new Tree(Node.createNode(indices,
      treeParameters, trainingData, new scala.util.Random(randomSeed)))
  }
}
