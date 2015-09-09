package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint

class RandomForest private (trees: List[Tree]) extends Serializable {
}

object RandomForest {
  def train(input: Iterator[LabeledPoint]): Iterator[RandomForest] = ???
}

