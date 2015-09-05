package SerialForest

import org.apache.spark.mllib.regression.LabeledPoint
import SerialForest.Tree

class RandomForest private (trees: List[Tree]) extends Serializable {
}

object RandomForest {
  //mapPartitions needs Iterator[T]) ⇒ Iterator[U], so to take and return an Iterator.
  //Adam: Using Iterators seems pretty annoying,
  //I guess if you need an array instead of an iterator you need to construct it here
  //Or maybe if you can find a work around, you can change it back.
  def train(input: Iterator[LabeledPoint]): Iterator[RandomForest] = ???
}

