package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint
import SerialForest.Tree

class RandomForest private (trees: List[Tree]) extends Serializable {
}

object RandomForest {
  def train(input: Array[LabeledPoint]): RandomForest = ???
}

